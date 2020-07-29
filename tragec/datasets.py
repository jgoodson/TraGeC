import random
from copy import copy
from pathlib import Path
from typing import Union, List, Any, Dict, Tuple, Callable
from functools import partial

import lmdb
import bson
import numpy as np
import torch
from tape.datasets import pad_sequences, dataset_factory
from torch.utils.data import Dataset

from .registry import registry


class GeCDataset(Dataset):

    def item_length(self, index):
        return 0

    @staticmethod
    def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, torch.Tensor]:
        pass


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 decode_method: Callable = bson.decode,
                 buffers=True):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = env.stat()['entries']

        self._env = env
        self._num_examples = num_examples
        self.decode_method = decode_method
        self.buffers = buffers

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: Union[int, str]):
        with self._env.begin(write=False, buffers=self.buffers) as txn:
            try:
                item = self.decode_method(txn.get(str(index).encode()))
            except TypeError:
                print(index, self._env.path())
        return item


@registry.register_task('masked_recon_modeling')
class MaskedReconstructionDataset(GeCDataset):
    """Creates the Masked Reconstruction Modeling RefSeq Dataset

    Args:
        data_path (Union[str, Path]): Path to tragec data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False,
                 seqvec_type: str = 'seqvec',
                 max_seq_len: int = 512):
        super().__init__()
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")

        data_path = Path(data_path)
        data_file = f'refseq/map{max_seq_len}/refseq_{split}.lmdb'
        refseq_file = f'refseq/refseq.lmdb'
        seqvec_file = f'seqvec/{seqvec_type}.lmdb'
        self.data = LMDBDataset(data_path / data_file, )
        self.refseq = LMDBDataset(data_path / refseq_file, )
        array_decode = partial(np.frombuffer, dtype=np.float32)
        self.seqvec = LMDBDataset(data_path / seqvec_file, decode_method=array_decode)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        refseq_id, locs = item['refseq'], item['indices']

        hashes, starts, stops, strands = zip(*self.refseq[refseq_id]['genes'][locs[0]:locs[1]])

        gene_reps = np.vstack([np.frombuffer(self.seqvec[h], dtype=np.float32) for h in hashes])

        masked_reps, targets = self._apply_pseudobert_mask(gene_reps)

        input_mask = np.ones(len(masked_reps))

        # TODO: mask these?
        strands = np.array(strands, dtype=np.int8)
        lengths = np.array(stops, dtype=np.int16) - np.array(starts, dtype=np.int16)

        return masked_reps, input_mask, targets, strands, lengths

    def item_length(self, index):
        item = self.data[index]
        refseq_id, locs = item['refseq'], item['indices']
        return locs[1] - locs[0]

    @staticmethod
    def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) \
            -> Dict[str, torch.Tensor]:
        gene_reps, input_mask, targets, strands, lengths = tuple(zip(*batch))

        gene_reps = torch.from_numpy(pad_sequences(gene_reps, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignores all-0 representations
        targets = torch.from_numpy(pad_sequences(targets, 0))
        # 1 = forward, -1 = reverse, 0 = pad
        strands = torch.from_numpy(pad_sequences(strands, 0))
        lengths = torch.from_numpy(pad_sequences(lengths, 0))

        return {'gene_reps': gene_reps,
                'input_mask': input_mask,
                'targets': targets,
                'strands': strands,
                'lengths': lengths}

    @staticmethod
    def _apply_pseudobert_mask(gene_reps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        masked_gene_reps = copy(gene_reps)
        rep_size = len(gene_reps[0])
        targets = np.zeros_like(masked_gene_reps)

        for i, gene_rep in enumerate(gene_reps):
            # Tokens begin and end with start_token and stop_token, ignore these

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                targets[i] = gene_rep

                if prob < 0.8:
                    # 80% random change to zero token
                    gene_rep = np.zeros(rep_size)
                elif prob < 0.9:
                    # 10% chance to change to random representation
                    gene_rep = np.random.normal(0, 1, rep_size)
                else:
                    # 10% chance to keep current representation
                    pass

                masked_gene_reps[i] = gene_rep

        return np.array(masked_gene_reps), targets


@registry.register_task('embed_gec')
class EmbedDataset(GeCDataset):

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):
        super().__init__()
        self.data = dataset_factory(data_file, in_memory=in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        gene_reps = item['primary']
        input_mask = np.ones(len(gene_reps))
        return item['id'], gene_reps, input_mask

    def item_length(self, index):
        return len(self[index])

    @staticmethod
    def collate_fn(batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        ids, gene_reps, input_mask = zip(*batch)
        ids = list(ids)
        gene_reps = torch.from_numpy(pad_sequences(gene_reps))
        input_mask = torch.from_numpy(pad_sequences(input_mask))
        return {'ids': ids, 'gene_reps': gene_reps, 'input_mask': input_mask}  # type: ignore
