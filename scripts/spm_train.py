import json
import os
from multiprocessing import cpu_count
from time import time

import click
import sentencepiece as spm
from tqdm import tqdm

PAD, UNK = "@pad@", "@@UNKNOWN@@"
START, END = "@start@", "@end@"


@click.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("model_prefix", type=click.STRING)
def spm_train(train_file,
              model_prefix):
    tmp_path = str(time())
    with open(tmp_path, "w") as file:
        for x in tqdm(map(json.loads, open(train_file)), desc="Prep"):
            print(x["text"], file=file)

    spm.SentencePieceTrainer.Train(input=tmp_path,
                                   model_prefix=model_prefix,
                                   model_type="bpe",
                                   vocab_size=32000,
                                   max_sentence_length=8192,
                                   character_coverage=1.,
                                   num_threads=cpu_count(),
                                   bos_piece=START,
                                   eos_piece=END,
                                   unk_piece=UNK,
                                   pad_piece=PAD,
                                   pad_id=0,
                                   bos_id=1,
                                   eos_id=2,
                                   unk_id=3)

    os.remove(tmp_path)
ø

if __name__ == '__main__':
    spm_train()
