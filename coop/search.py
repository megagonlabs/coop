import json
import shutil
import tarfile
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Union

import click
import pandas as pd
import rouge
import torch

from coop.models import Model
from coop.reader import ReviewTest, OptimusTest
from coop.tokenizer import Tokenizer
from coop.util import avg, overlap, oracle, load_tokenizer, load_data, build_model, BAD_WORDS, powerset


def brute_force_gen(model: Model,
                    data: Union[ReviewTest, OptimusTest],
                    tgt_tokenizer: Tokenizer,
                    num_beams: int = 4,
                    bad_words_ids: List[int] = None,
                    split: int = 1, ):
    outs = []
    for i, x in enumerate(data):
        z_raw = model(**x).q.loc
        idxes = powerset(z_raw.size(0))
        zs = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes])
        gens = []
        for z in torch.split(zs, len(idxes) // split):
            g = model.generate(z, num_beams=num_beams, bad_words_ids=bad_words_ids)
            gens.extend(tgt_tokenizer.decode(g))
        outs.append([{"selected": [x["reviews"][i] for i in idx],
                      "reviews": x["reviews"],
                      "summary": x["summary"],
                      "predicted": gen,
                      "idx": idx} for idx, gen in zip(idxes, gens)])
    return outs


@click.command()
@click.argument("log_dir_or_file", type=click.Path(exists=True))
@click.option("--split", type=click.INT, default=1)
def main(log_dir_or_file, split):
    log_dir_or_file = Path(log_dir_or_file)
    tempdir = None
    if not log_dir_or_file.is_dir():
        # Extract archive
        tempdir = tempfile.mkdtemp()
        with tarfile.open(log_dir_or_file, "r:gz") as archive:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(archive, tempdir)
        log_dir = Path(tempdir)
    else:
        log_dir = Path(log_dir_or_file)

    config = json.load(open(log_dir / "config.json"))
    src_tokenizer, tgt_tokenizer = load_tokenizer(config)
    _, dev, test = load_data(config, src_tokenizer, tgt_tokenizer)
    bad_words_ids = tgt_tokenizer.get_ids(BAD_WORDS)
    if config["model"]["type"].lower() == "bimeanvae" and config["data_dir"].endswith("amzn"):
        # The amzn dataset often includes the pronoun I without prefix. To avoid the issue, this tweak is applied.
        bad_words_ids.extend(tgt_tokenizer.get_ids("I", no_prefix=True))
    model = build_model(config).eval()

    model.load_state_dict(torch.load(log_dir / "pytorch_model.bin", map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        model.cuda()

    dev_gen = brute_force_gen(model, dev, tgt_tokenizer, bad_words_ids=bad_words_ids, split=split)
    test_gen = brute_force_gen(model, test, tgt_tokenizer, bad_words_ids=bad_words_ids, split=split)

    coop = {}
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True)

    with Pool(cpu_count()) as p:
        for func in (avg, overlap, oracle):
            name = func.__name__
            coop[name] = {}
            for key, val in (("dev", dev_gen), ("test", test_gen)):
                coop_score = p.map(func, val)
                index = list(range(len(val[0])))
                index = [max(index, key=lambda x: s[x]) for s in coop_score]
                selected = [v[i] for i, v in zip(index, val)]
                rouge_score = evaluator.get_scores(
                    [x["predicted"] for x in selected], [x["summary"] for x in selected])
                rouge_score = {"_".join((metric, k)): v for metric, vs in rouge_score.items() for k, v in
                               vs.items()}
                coop[name][key] = {
                    "coop_score": coop_score,
                    "index": index,
                    "rouge": rouge_score}

            df = pd.DataFrame({k: coop[name][k]["rouge"] for k in ("dev", "test")})
            df.sort_index(inplace=True)
            print(name)
            print(df)

    # Clean-up
    if tempdir is not None:
        shutil.rmtree(tempdir, ignore_errors=True)
    else:
        json.dump(coop, open(log_dir / "coop.json", "w"))


if __name__ == '__main__':
    main()
