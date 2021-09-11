import json
import linecache
from pathlib import Path
from typing import List

import torch.tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


def get_length(fp):
    i = 0
    for i, _ in enumerate(open(fp), 1):
        pass
    return i


class ReviewDataset(Dataset):
    def __init__(self,
                 data_file,
                 tokenizer: Tokenizer):
        super().__init__()
        assert Path(data_file).exists(), f"Data directory, {data_file}, does not exist."
        self.data_file = str(data_file)
        self.tokenizer = tokenizer

        self.len = get_length(data_file)

    def __getitem__(self, idx):
        ins = json.loads(linecache.getline(self.data_file, idx + 1))
        x = [self.tokenizer.bos_id] + ins["piece"] + [self.tokenizer.eos_id]
        return torch.tensor(x), ins["text"]

    def __len__(self):
        return self.len

    def collate_fn(self, data: List[torch.Tensor]):
        tensor, reviews = zip(*data)
        tensor = pad_sequence(tensor, batch_first=True, padding_value=self.tokenizer.pad_id)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return {"src": tensor, "tgt": tensor, "reviews": reviews}


class OptimusDataset(Dataset):
    def __init__(self,
                 data_file,
                 src_tokenizer: Tokenizer,
                 tgt_tokenizer: Tokenizer):
        super().__init__()
        assert Path(data_file).exists(), f"Data directory, {data_file}, does not exist."
        self.data_file = str(data_file)
        self.pad, self.bos, self.eos = '<PAD>', '<BOS>', '<EOS>'

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = self.tokenizer = tgt_tokenizer

        self.len = get_length(data_file)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = json.loads(linecache.getline(self.data_file, idx + 1))["text"]
        return x

    def collate_fn(self, data: List[str]):
        src = self.src_tokenizer(data,)
        tgt = self.tgt_tokenizer(data)
        return {"src": src, "tgt": tgt, "reviews": data}


class ReviewTest(Dataset):
    def __init__(self,
                 data_file,
                 tokenizer: Tokenizer):
        super().__init__()
        assert Path(data_file).exists(), f"Data directory, {data_file}, does not exist."
        self.data = json.load(open(data_file))
        self.tokenizer = tokenizer

        self.len = len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx]
        reviews = ins["reviews"]
        summary = ins["summary"]
        tensor = self.tokenizer(ins["reviews"])
        return {"src": tensor, "reviews": reviews, "summary": summary}

    def __len__(self):
        return self.len


class OptimusTest(Dataset):
    def __init__(self,
                 data_file,
                 src_tokenizer: Tokenizer,
                 tgt_tokenizer: Tokenizer):
        super().__init__()
        assert Path(data_file).exists(), f"Data directory, {data_file}, does not exist."
        self.data = json.load(open(data_file))
        self.pad, self.bos, self.eos = '<PAD>', '<BOS>', '<EOS>'

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = self.tokenizer = tgt_tokenizer

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ins = self.data[idx]
        src = self.src_tokenizer(ins["reviews"])
        return {"src": src, "reviews": ins["reviews"], "summary": ins["summary"]}
