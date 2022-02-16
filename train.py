import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Dict, List

import click
import pandas as pd
import torch
from _jsonnet import evaluate_file
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from coop.models import Model, BiMeanVAE, Optimus
from coop.util import get_logger, load_tokenizer, load_data, build_model
from evaluate import evaluate


class Trainer:
    def __init__(self,
                 model: Model,
                 data: List[Dataset],
                 log_dir: Path,
                 num_steps: int,
                 checkout_step: int,
                 batch_size: int,
                 lr: float = 1e-4,
                 clip_value: float = 5.,
                 max_norm: float = 1.,
                 num_keep: int = 10):
        log_dir = Path(log_dir)
        if torch.cuda.is_available():
            model.cuda()

        self.model = model
        self.train, self.dev, self.test = data

        self.opt = Adam(self.model.parameters(), lr, betas=(0.5, 0.999), eps=1e-6, )

        self.scheduler = get_linear_schedule_with_warmup(self.opt, checkout_step // 10, num_steps)

        self.clip_value = clip_value
        self.max_norm = max_norm
        self.num_steps = num_steps
        self.checkout_step = checkout_step
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.logger = get_logger(log_dir)
        self.losses = defaultdict(list)
        self.best_score = 0.
        self.writer = {key: SummaryWriter(log_dir=str(log_dir / "log" / key)) for key in ("train", "dev", "test")}
        self.global_step = 0
        self.num_keep = num_keep
        self.model_path = []

    @classmethod
    def from_config(cls,
                    config: dict,
                    log_dir: Path):
        json.dump(config, open(log_dir / "config.json", "w"))
        if "spm_path" in config:
            shutil.copy(config["spm_path"], log_dir / "spm.model")
        tokenizers = load_tokenizer(config)
        data = load_data(config, *tokenizers)
        model = build_model(config)

        return cls(model, data, log_dir, **config.pop("trainer"))

    def _fit_partial(self,
                     batch,
                     p: tqdm = None):
        self.model.train()
        self.model.zero_grad()
        losses = self.model(**batch)
        nll, zkl, zkl_real = losses.nll, losses.zkl, losses.zkl_real
        klw = self.model.klw(self.global_step, self.checkout_step)
        loss = nll + klw * zkl
        loss.backward()
        if isinstance(self.model, Optimus):
            clip_grad_norm_(self.model.parameters(), self.max_norm)
        else:
            clip_grad_value_(self.model.parameters(), self.clip_value)

        loss_dict = {"nll": nll.item(), "klw": klw, "zkl": zkl.item(), "zkl_real": zkl_real.item()}

        self.opt.step()
        self.scheduler.step()

        if p is not None:
            for k, v in loss_dict.items():
                self.writer["train"].add_scalar(f"Loss/{k}", v, global_step=self.global_step)
                self.losses[k].append(v)
            p.set_postfix(**loss_dict)
            p.update()

    def fit(self):
        train = DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.train.collate_fn)
        p = tqdm(desc=f"Step {self.global_step}", total=self.checkout_step, ncols=100)

        while True:
            for batch in train:
                self.global_step += 1
                self._fit_partial(batch, p=p)
                if self.global_step % self.checkout_step == 0:
                    losses = self._avg_loss(p)
                    self._archive(losses)
                    p.close()
                    self._evaluate()
                    if isinstance(self.model, BiMeanVAE) and self.global_step == 10000:
                        self.logger.info("Reset LSTM decoder")
                        self.model.decoder.reset_parameters()
                    if self.global_step == self.num_steps:
                        self._finalize()
                        return
                    p = tqdm(desc=f"Step {self.global_step}", total=self.checkout_step, ncols=100)

    def _finalize(self):
        archive_file = self.log_dir / "model.tar.gz"
        with tarfile.open(archive_file, "w:gz") as archive:
            archive.add(self.log_dir / "config.json", arcname="config.json")
            archive.add(self.log_dir / "best.th", arcname="pytorch_model.bin")
            if isinstance(self.model, BiMeanVAE):
                archive.add(self.log_dir / "spm.model", arcname="spm.model")

    def _evaluate(self):
        self.model.eval()
        # Summarize
        metrics = {}
        for data_type in ("dev", "test"):
            data = getattr(self, data_type)
            metrics[data_type] = evaluate(self.model, data, debug=True)
            for k, v in metrics[data_type].items():
                metric, tgt, key = k.split("_")
                self.writer[data_type].add_scalar(f"Metrics/{tgt}/{metric}/{key}/", v, global_step=self.global_step)

        df = pd.DataFrame(metrics)
        df.sort_index(inplace=True)
        print(df)
        json.dump(metrics, open(self.log_dir / f"metrics-step_{self.global_step}.json", "w"))
        dev_scores = {f"R{i}": df["dev"][f"rouge-{i}_sum_f"] for i in "12l"}
        if sum(dev_scores.values()) > self.best_score:
            self.best_score = sum(dev_scores.values())
            shutil.copy(self.log_dir / f"metrics-step_{self.global_step}.json", self.log_dir / "metrics.json")
            shutil.copy(self.log_dir / f"model-step_{self.global_step}.th", self.log_dir / "best.th")
            shutil.copy(self.log_dir / f"training_metrics-step_{self.global_step}.json",
                        self.log_dir / "training_metrics.json")
            self.logger.info("Best scores")
            for k, v in dev_scores.items():
                self.logger.info(f"DEV: {k}={100 * v:.2f}")
            test_scores = {f"R{i}": df["test"][f"rouge-{i}_sum_f"] for i in (1, 2, "l")}
            for k, v in test_scores.items():
                self.logger.info(f"TEST: {k}={100 * v:.2f}")

    def _archive(self,
                 losses: Dict[str, float]):
        model_path = self.log_dir / f"model-step_{self.global_step}.th"
        torch.save(self.model.state_dict(), model_path)
        json.dump(losses, open(self.log_dir / f"training_metrics-step_{self.global_step}.json", "w"))
        self.model_path.append(model_path)
        if len(self.model_path) > self.num_keep:
            self.model_path.pop(0).unlink()

    def _avg_loss(self,
                  p: tqdm):
        losses = {k: sum(v) / len(v) for k, v in self.losses.items()}
        losses["klw"] = 1.
        p.set_postfix(**losses)
        p.update()
        self.losses.clear()
        return losses


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--log_dir", "-s", type=click.Path(), default=f"/tmp/{str(int(time()))}")
def main(config_file, log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True)

    config = json.loads(evaluate_file(config_file))

    trainer = Trainer.from_config(config, log_dir)
    trainer.fit()


if __name__ == '__main__':
    main()
