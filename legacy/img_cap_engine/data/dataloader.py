import os
import json
from typing import Callable, List, Tuple
from itertools import groupby
from random import shuffle

import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import AutoTokenizer


class DataLoader(object):
    def __init__(
        self,
        data_dir: str,
        img_dir: str,
        split: str,
        tok_in_batch: int,
        transforms: Callable[[Image.Image], torch.Tensor],
        max_b_size: int,
    ):
        """
        Initialize the data loader.

        Parameters:
        -----------
        data_dir : str
            Directory containing the JSON file with image and caption data.
        img_dir : str
            Directory containing the images.
        split : str
            Data split ('train', 'val', 'test', 'tiny_train', 'tiny_val', 'tiny_test').
        tok_in_batch : int
            Maximum number of tokens per batch.
        transforms : Callable[[Image.Image], torch.Tensor]
            Transform function to apply to images.
        max_b_size : int
            Maximum batch size.
        """

        self.tok_in_batch: int = tok_in_batch
        self.transforms: Callable = transforms
        self.max_b_size: int = max_b_size
        self.split: str = split.lower()
        assert self.split in {
            "train",
            "val",
            "test",
            "tiny_train",
            "tiny_val",
            "tiny_test",
        }, "'split' must be in ['train', 'val', 'test', 'tiny_train', 'tiny_val', 'tiny_test']"
        self.is_train: bool = self.split == "train" or self.split == "tiny_train"

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(
            {"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>"}
        )

        file_path = os.path.join(data_dir, f"{self.split}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imgs, caps = [], []
        for item in data:
            caps.append(item["cap"])
            imgs.append(item["img"])

        caps_len = [len(self.tokenizer.encode("<SOS>" + cap + "<EOS>")) for cap in caps]
        imgs = [os.path.join(img_dir, img) for img in imgs]

        self.data: List[Tuple[str, str, int]] = list(zip(imgs, caps, caps_len))

        if self.is_train:
            self.data.sort(key=lambda x: x[2])

        self.create_batches()

    def create_batches(self):
        if self.is_train:
            chunks = [list(t) for _, t in groupby(self.data, key=lambda x: x[2])]
            self.batches = list()
            for chunk in chunks:  # (n) x [img, cap, cap_len]
                # tok_in_batch // max(cap_len) ==> How many max(chunk) = len(chunk[0][3]) in tok_in_batch
                item_in_batch = min(self.tok_in_batch // chunk[0][2], self.max_b_size)
                # chunk -> chunk1, chunk2... ==> batches
                self.batches.extend(
                    [
                        chunk[i : i + item_in_batch]
                        for i in range(0, len(chunk), item_in_batch)
                    ]
                )

            shuffle(self.batches)
            self.n_batches = len(self.batches)
            self.idx = -1
        else:
            self.batches = [[ic] for ic in self.data]
            self.n_batches = len(self.batches)
            self.idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        try:
            img, cap, cap_len = zip(*self.batches[self.idx])
        except IndexError:
            raise StopIteration

        img = [self.transforms(Image.open(i).convert("RGB")) for i in img]
        cap = [self.tokenizer.encode("<SOS>" + c + "<EOS>") for c in cap]

        img = torch.stack(img)
        cap = pad_sequence(
            sequences=[torch.LongTensor(c) for c in cap],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        cap_len = torch.LongTensor(cap_len)

        return img, cap, cap_len
