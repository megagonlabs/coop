import logging
import sys
from itertools import chain, combinations
from pathlib import Path

import rouge

from .models import BiMeanVAE, Optimus
from .reader import ReviewDataset, ReviewTest, OptimusDataset, OptimusTest
from .tokenizer import Tokenizer, SpmTokenizer, BERTTokenizer, GPT2Tokenizer

R1 = rouge.Rouge(metrics=["rouge-n"], max_n=1, limit_length=False, apply_avg=True, stemming=True,
                 ensure_compatibility=True)
BAD_WORDS = ["I", "i", "My", "my", "Me", "me", "We", "we", "Our", "our", "us"]


def powerset(size):
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    return list(map(list, chain.from_iterable(combinations(range(size), r + 1) for r in range(size))))


def get_logger(log_dir: Path):
    fmt = "'%(asctime)s - %(levelname)s - %(name)s -   %(message)s'"
    datefmt = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    file = logging.FileHandler(log_dir / "logging")
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    logger.addHandler(file)

    return logger


def load_tokenizer(config: dict):
    model_type = config["model"]["type"]
    if model_type == "optimus":
        src_tokenizer = BERTTokenizer(config.get("device"))
        tgt_tokenizer = GPT2Tokenizer(config.get("device"))
    else:
        src_tokenizer = tgt_tokenizer = SpmTokenizer(config["spm_path"], config.get("device"))
        config["model"]["vocab_size"] = src_tokenizer.vocab_size
    config["model"].update({"pad_id": tgt_tokenizer.pad_id,
                            "bos_id": tgt_tokenizer.bos_id,
                            "eos_id": tgt_tokenizer.eos_id})
    return src_tokenizer, tgt_tokenizer


def load_data(config: dict, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer = None):
    model_type = config["model"]["type"]
    data_dir = Path(config["data_dir"])
    if model_type == "optimus":
        train = OptimusDataset(data_dir / "train.jsonl", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        dev = OptimusTest(data_dir / "dev.json", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        test = OptimusTest(data_dir / "test.json", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    else:
        train = ReviewDataset(data_dir / "train.jsonl", tokenizer=src_tokenizer)
        dev = ReviewTest(data_dir / "dev.json", tokenizer=src_tokenizer)
        test = ReviewTest(data_dir / "test.json", tokenizer=src_tokenizer)

    return train, dev, test


def build_model(config: dict):
    model_type = config["model"].pop("type").lower()
    if model_type == "bimeanvae":
        cls = BiMeanVAE
    elif model_type == "optimus":
        cls = Optimus
    else:
        raise ValueError(f"Model type {model_type} is not available.")
    return cls(**config.pop("model"))


def avg(ins):
    return [len(x["selected"]) for x in ins]


def overlap(ins):
    scores = []
    for i in ins:
        s = R1.get_scores(i["predicted"], i["reviews"])
        scores.append(s["rouge-1"]["f"])
    return scores


def oracle(ins):
    scores = []
    for i in ins:
        s = R1.get_scores([i["predicted"]], [i["summary"]])
        scores.append(s[f"rouge-1"]["f"])
    return scores


def input_output_overlap(inputs, output):
    r1 = rouge.Rouge(metrics=["rouge-n"], max_n=1, limit_length=False,)
    return r1.get_scores(output, inputs)["rouge-1"]["f"]
