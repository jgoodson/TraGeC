import math
import typing

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

from ..datasets import BioDataset
from ..utils._sampler import BucketBatchSampler


class SimpleMLP(nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=0),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def get_activation_fn(name: str) -> typing.Callable:
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")


class SimpleConv(nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim),  # Added this
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x


class BioDataModule(pl.LightningDataModule):
    prefix = ''

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 max_seq_len: int,
                 num_workers: int,
                 seqvec_type: typing.Optional[str] = None,
                 tokenizer: typing.Optional[str] = None,
                 xla: bool = False,
                 **opt_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.seqvec_type = seqvec_type
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.distributed = False
        self.splits = {}
        self.test_split = ''
        self.train_split = ''
        self.val_split = ''
        self.xla = xla
        for kwarg, val in opt_kwargs.items():
            self.__setattr__(kwarg, val)

    def setup(self, stage: typing.Optional[str] = None) -> None:
        # TODO: process additional kwargs for datasets appropriately
        self.splits = {
            split: self.dataset(
                data_path=self.data_dir,
                split=split,
            )
            for split in self.split_names
        }

    def _prep_loader(self, dataset: BioDataset, test_only: bool) -> DataLoader:
        if test_only or not self.distributed:
            sampler = RandomSampler(dataset)
        elif self.xla:
            import torch_xla.core.xla_model as xm
            sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        else:
            sampler = DistributedSampler(dataset, )

        batch_sampler = BucketBatchSampler(
            sampler, self.batch_size, False, dataset, 10)

        loader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            batch_sampler=batch_sampler,
        )
        return loader

    def get_dataloader(self, split: str, test_only: bool = False) -> DataLoader:
        return self._prep_loader(self.splits[split], test_only)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if not self.train_split:
            raise AttributeError('train_split attribute has not been set for DataModule subclass')
        return self.get_dataloader(self.train_split, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        if not self.val_split:
            raise AttributeError('val_split attribute has not been set for DataModule subclass')
        return self.get_dataloader(self.val_split, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        if not self.test_split:
            raise AttributeError('test_split attribute has not been set for DataModule subclass')
        return self.get_dataloader(self.test_split, **kwargs)
