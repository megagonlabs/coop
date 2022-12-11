import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn

from .util import load_tokenizer, build_model

AVAILABLE_MODELS = {"megagonlabs/bimeanvae-yelp",
                    "megagonlabs/bimeanvae-amzn",
                    "megagonlabs/optimus-yelp",
                    "megagonlabs/optimus-amzn"}


class VAE(nn.Module):
    def __init__(self, model_name_or_path: str, device: str = None):
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Find the model path
        if Path(model_name_or_path).exists():
            tempdir = tempfile.mkdtemp()
            try:
                # Extract archive
                with tarfile.open(model_name_or_path, "r:gz") as archive:
                    
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
                model_dir = Path(tempdir)
                # Load model
                config = json.load(open(model_dir / "config.json"))
                config["device"] = self.device
                model_path = model_dir / "pytorch_model.bin"
                state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

            finally:
                # Clean-up
                shutil.rmtree(tempdir, ignore_errors=True)

        else:
            assert str(model_name_or_path) in AVAILABLE_MODELS, AVAILABLE_MODELS
            # Lazy import
            from huggingface_hub import hf_hub_url, cached_download
            config_url = hf_hub_url(str(model_name_or_path), filename="config.json")
            config = json.load(open(cached_download(url=config_url, library_name="coop")))
            model_url = hf_hub_url(str(model_name_or_path), filename="pytorch_model.bin")
            model_path = cached_download(url=model_url, library_name="coop")
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

            if "bimeanvae" in str(model_name_or_path):
                spm_url = hf_hub_url(str(model_name_or_path), filename="spm.model")
                spm_path = cached_download(url=spm_url, library_name="coop")
                config["spm_path"] = spm_path

        self.src_tokenizer, self.tgt_tokenizers = load_tokenizer(config)
        self.model = build_model(config).eval()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self,
               reviews: Union[List[str], str],
               device: str = None):
        if isinstance(reviews, str):
            reviews = [reviews]

        if device is None:
            self.to(self.device)
        src = self.src_tokenizer(reviews)
        return self.model(src).q.loc

    @torch.no_grad()
    def generate(self,
                 z: torch.Tensor,
                 num_beams: int = 4,
                 max_tokens: int = 256,
                 bad_words: Union[str, List[str], List[int]] = None):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        if bad_words is not None:
            if isinstance(bad_words, str):
                bad_words = [bad_words]
            if isinstance(bad_words[0], str):
                bad_words_ids = self.tgt_tokenizers.get_ids(bad_words)
            else:
                bad_words_ids = bad_words
        else:
            bad_words_ids = None

        return self.tgt_tokenizers.decode(self.model.generate(
            z=z,
            num_beams=num_beams,
            max_tokens=max_tokens,
            bad_words_ids=bad_words_ids))
