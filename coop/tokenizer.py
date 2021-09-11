from typing import List, Union
from sentencepiece import SentencePieceProcessor

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizerFast, GPT2TokenizerFast


class Tokenizer:
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def __call__(self, reviews: Union[List[str], str]):
        raise NotImplementedError

    @property
    def bos_id(self):
        raise NotImplementedError

    @property
    def eos_id(self):
        raise NotImplementedError

    @property
    def pad_id(self):
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError

    def decode(self,
               ids: Union[List[List[int]], torch.Tensor]):
        raise NotImplementedError


class SpmTokenizer(Tokenizer):
    def __init__(self, spm_path: str, device: str = None):
        super().__init__(device)
        self.spm = SentencePieceProcessor()
        self.spm.Load(spm_path)

    def __call__(self, reviews: Union[List[str], str]):
        if isinstance(reviews, str):
            reviews = [reviews]
        tensor = [torch.tensor([self.bos_id] + self.spm.Encode(r) + [self.eos_id]) for r in reviews]
        tensor = pad_sequence(tensor, batch_first=True, padding_value=self.pad_id)
        tensor = tensor.to(self.device)
        return tensor

    @property
    def bos_id(self):
        return self.spm.bos_id()

    @property
    def eos_id(self):
        return self.spm.eos_id()

    @property
    def pad_id(self):
        return self.spm.pad_id()

    @property
    def vocab_size(self):
        return self.spm.GetPieceSize()

    def get_ids(self, pieces: List[str], no_prefix: bool = False):
        if not no_prefix:
            pieces = ["▁" + p for p in pieces]
        return [self.spm.PieceToId(p) for p in pieces]

    def decode(self,
               ids: Union[List[List[int]], torch.Tensor]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self.spm.DecodeIdsWithCheck(x) for x in ids]


class BERTTokenizer(Tokenizer):
    def __init__(self, device: str = None):
        super().__init__(device)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    def __call__(self, reviews: Union[List[str], str]):
        if isinstance(reviews, str):
            reviews = [reviews]
        src = self.tokenizer(reviews, padding=True, add_special_tokens=True, truncation=True, max_length=256,
                             return_tensors="pt", )
        src = {k: v.to(self.device) for k, v in src.items()}
        return src

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id


class GPT2Tokenizer(Tokenizer):
    def __init__(self, device: str = None):
        super().__init__(device)
        self.pad, self.bos, self.eos = '<PAD>', '<BOS>', '<EOS>'
        sp = {'pad_token': self.pad, 'bos_token': self.bos, 'eos_token': self.eos}
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(sp)

    def __call__(self, reviews: Union[List[str], str]):
        if isinstance(reviews, str):
            reviews = [reviews]
        tgt = self.tokenizer([" ".join((self.bos, x, self.eos)) for x in reviews],
                             padding=True, truncation=True, max_length=256, return_tensors="pt")
        tgt["labels"] = tgt["input_ids"]
        del tgt["attention_mask"]
        tgt = {k: v.to(self.device) for k, v in tgt.items()}
        return tgt

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def get_ids(self, pieces: List[str]):
        ids = self.tokenizer.convert_tokens_to_ids(pieces)
        ids += self.tokenizer.convert_tokens_to_ids(["Ġ" + w for w in pieces])
        return [[w] for w in ids]

    def decode(self,
               ids: Union[List[List[int]], torch.Tensor]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [x.strip() for x in self.tokenizer.batch_decode(ids, skip_special_tokens=True,)]
