# From songlab-cal TAPE: https://github.com/songlab-cal/tape
# Extended and modified for TraGeC

import math
import pickle
import random
from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from pathlib import Path
from typing import Union, List, Any, Dict, Tuple, Callable, Sequence, Sized, Optional

import lmdb
import bson
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from .tokenizers import TAPETokenizer


def roundup(n, f=8):
    return n + (f - n) % f


def pad_sequences(sequences: Sequence[Union[np.ndarray, torch.Tensor]],
                  constant_value: int = 0,
                  dtype: Optional[Union[np.dtype, torch.dtype]] = None) -> np.ndarray:
    batch_size = len(sequences)
    # noinspection PyTypeChecker
    shape = [batch_size] + np.max([tuple(roundup(s) for s in seq.shape) for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)
    else:
        raise TypeError('Sequence value to pad not ndarray or Tensor')

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class SizedDataset(Dataset, Sized, ABC):
    pass


class BioDataset(SizedDataset):

    def item_length(self, index) -> int:
        return 0

    @staticmethod
    @abstractmethod
    def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, torch.Tensor]:
        pass


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> SizedDataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    # elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
    #    return FastaDataset(data_file, *args, **kwargs)
    # elif data_file.suffix == '.json':
    #    return JSONDataset(data_file, *args, **kwargs)
    # elif data_file.is_dir():
    #    return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")


class LMDBDataset(SizedDataset):
    """Creates a datamodule from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        decode_method (Callable): Function with which to decode individual items
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 decode_method: Callable = pickle.loads):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        self._data_file = str(data_file)
        env = lmdb.open(self._data_file, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            if txn.get(b'num_examples'):
                num_examples = decode_method(txn.get(b'num_examples'))
            else:
                num_examples = env.stat()['entries']

        self._env = env
        self._num_examples = num_examples
        self.decode_method = decode_method

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: Union[int, str]):
        with self._env.begin(write=False) as txn:
            item = self.decode_method(txn.get(str(index).encode()))

        return item

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_env']
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = lmdb.open(self._data_file, max_readers=1, readonly=True,
                              lock=False, readahead=False, meminit=False)


class GeCMaskedReconstructionDataset(BioDataset):
    """Creates the Masked Reconstruction Modeling RefSeq Dataset

    Args:
        data_path (Union[str, Path]): Path to tragec data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
    """

    def __init__(self, data_path: Union[str, Path], split: str, seqvec_type: str = 'seqvec', max_seq_len: int = 512,
                 percentmasked=.15, dtype=np.float32, **kwargs):
        super().__init__()
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")

        data_path = Path(data_path)
        data_file = f'refseq/maps{max_seq_len}/refseq_{split}.lmdb'
        refseq_file = f'refseq/refseq.lmdb'
        seqvec_file = f'seqvec/{seqvec_type}.lmdb'
        self.data = LMDBDataset(data_path / data_file, decode_method=bson.decode)
        self.refseq = LMDBDataset(data_path / refseq_file, decode_method=bson.decode)
        array_decode = partial(np.frombuffer, dtype=dtype)
        self.seqvec = LMDBDataset(data_path / seqvec_file, decode_method=array_decode)
        self.percentmasked = percentmasked

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        refseq_id, locs = item['refseq'], item['indices']

        # Data includes forward and reverse strand indices, manage reverse strand properly
        if locs[1] > locs[0]:
            hashes, starts, stops, strands = zip(*self.refseq[refseq_id]['genes'][locs[0]:locs[1]])
        else:
            hashes, starts, stops, strands = zip(*self.refseq[refseq_id]['genes'][locs[1]:locs[0]][::-1])

        # TODO mask strands/lengths?
        lengths = np.array(stops, dtype=np.int16) - np.array(starts, dtype=np.int16)
        strands = np.array(strands, dtype=np.int8)

        # Strand +/- is arbitrary based on sequencing, only relative strand matters, randomize each time
        if np.random.random() > 0.5:
            strands *= -1

        gene_reps = np.vstack([np.frombuffer(self.seqvec[h], dtype=np.float32) for h in hashes])

        masked_reps, targets = self._apply_pseudobert_mask(gene_reps, self.percentmasked)

        input_mask = np.ones(len(masked_reps))

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

        return {'sequence_rep': gene_reps,
                'input_mask': input_mask,
                'targets': targets,
                'strands': strands,
                'lengths': lengths}

    @staticmethod
    def _apply_pseudobert_mask(gene_reps: np.ndarray, percentmasked=.15) -> Tuple[np.ndarray, np.ndarray]:
        masked_gene_reps = copy(gene_reps)
        rep_size = len(gene_reps[0])
        num_genes = gene_reps.shape[0]
        targets = np.zeros_like(masked_gene_reps)

        num_masked = math.ceil(percentmasked * num_genes)
        mask_array = np.array([1] * num_masked + [0] * (num_genes - num_masked))
        np.random.shuffle(mask_array)
        for i, gene_rep in enumerate(gene_reps):

            prob = random.random()
            if mask_array[i] == 1:
                targets[i] = gene_rep

                if prob < 0.8:
                    # 80% random change to zero vector
                    gene_rep = np.zeros(rep_size)
                elif prob < 0.9:
                    # 10% chance to change to random representation
                    gene_rep = np.random.normal(0, 1, rep_size)
                else:
                    # 10% chance to keep current representation
                    pass

                masked_gene_reps[i] = gene_rep

        return np.array(masked_gene_reps), targets


class EmbedDataset(BioDataset):
    # TODO: find out if this actually works or if I test it anywhere

    def __init__(self,
                 data_file: Union[str, Path],
                 **kwargs):
        super().__init__()
        self.data = dataset_factory(data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        gene_reps = item['primary']
        input_mask = np.ones(len(gene_reps))
        return item['id'], gene_reps, input_mask

    def item_length(self, index):
        return len(self[index]['primary'])

    @staticmethod
    def collate_fn(batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        ids, gene_reps, input_mask = zip(*batch)
        ids = list(ids)
        gene_reps = torch.from_numpy(pad_sequences(gene_reps))
        input_mask = torch.from_numpy(pad_sequences(input_mask))
        return {'ids': ids, 'sequence_rep': gene_reps, 'input_mask': input_mask}  # type: ignore


class GeCClassificationDataset(BioDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 *args,
                 **kwargs):
        super().__init__()

        if split not in ('train', 'valid'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid']")

        data_path = Path(data_path)

        data_file = f'axen/bmcs_{split}.lmdb'
        self.data = LMDBDataset(data_path / data_file, decode_method=bson.decode)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        gene_reps = np.vstack([np.frombuffer(v, dtype=np.float32) for v in item['Protein Vectors']])
        input_mask = np.ones(len(gene_reps))
        strands = np.array([1 if x == '+' else -1 for x in item['Protein Strands']])
        lengths = np.array(item['Protein Lengths'])
        return gene_reps, input_mask, item['gec_type'], strands, lengths

    def item_length(self, index: int):
        item = self.data[index]
        return len(item['Protein Vectors'])

    @staticmethod
    def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) \
            -> Dict[str, torch.Tensor]:
        with open('test', 'w') as o:
            o.write(str(batch))
        gene_reps, input_mask, targets, strands, lengths = tuple(zip(*batch))

        gene_reps = torch.from_numpy(pad_sequences(gene_reps, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        targets = torch.LongTensor(targets)
        strands = torch.from_numpy(pad_sequences(strands, 0))
        lengths = torch.from_numpy(pad_sequences(lengths, 0))

        return {'sequence_rep': gene_reps,
                'targets': targets,
                'input_mask': input_mask,
                'strands': strands,
                'lengths': lengths}


class ProteinMaskedLanguageModelingDataset(BioDataset):
    """Creates the Masked Language Modeling Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 percentmasked: float = .15):
        super().__init__()
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        if data_path:
            data_path = Path(data_path)
            data_file = f'pfam/pfam_{split}.lmdb'
            self.data = dataset_factory(data_path / data_file)
        self.percentmasked = percentmasked

    def __len__(self) -> int:
        return len(self.data)

    def item_length(self, index):
        return len(self.data[index]['primary'])

    def __getitem__(self, index):
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        return masked_token_ids, input_mask, labels, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        # The following data is available from TAPE
        # clan = torch.LongTensor(clan)  # type: ignore
        # family = torch.LongTensor(family)  # type: ignore

        return {'sequence_rep': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue

            prob = random.random()
            if prob < self.percentmasked:
                prob /= self.percentmasked
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels


class ProteinRemoteHomologyDataset(BioDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac', ):

        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'remote_homology/remote_homology_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, item['fold_label']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fold_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fold_label = torch.LongTensor(fold_label)  # type: ignore

        return {'sequence_rep': input_ids,
                'input_mask': input_mask,
                'targets': fold_label}


class ProteinSecondaryStructureDataset(BioDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac', ):

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, -1))

        output = {'sequence_rep': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


class ProteinFluorescenceDataset(BioDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac', ):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['log_fluorescence'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
        fluorescence_true_value = fluorescence_true_value.unsqueeze(1)

        return {'sequence_rep': input_ids,
                'input_mask': input_mask,
                'targets': fluorescence_true_value}


class ProteinStabilityDataset(BioDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac', ):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'stability/stability_{split}.lmdb'

        self.data = dataset_factory(data_path / data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['stability_score'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, stability_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
        stability_true_value = stability_true_value.unsqueeze(1)

        return {'sequence_rep': input_ids,
                'input_mask': input_mask,
                'targets': stability_true_value}


class ProteinnetDataset(BioDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac', ):

        if split not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        protein_length = len(item['primary'])
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        valid_mask = item['valid_mask']
        contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)

        yind, xind = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        return token_ids, input_mask, contact_map, protein_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, contact_labels, protein_length = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, -1))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'sequence_rep': input_ids,
                'input_mask': input_mask,
                'targets': contact_labels,
                'protein_length': protein_length}


class ProteinDomainDataset(BioDataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac'):
        # need to change the labels by looking at the pfam data
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'holdout',")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'domain/domain_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, item['domains']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, family_label = tuple(zip(*batch))

        family_label_multihot = np.zeros((len(family_label), 14808))
        for idx, label in enumerate(family_label):
            for elem in label:
                family_label_multihot[idx][elem] = 1

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        family_label = torch.from_numpy(family_label_multihot)  # type: ignore

        return {'sequence_rep': input_ids,
                'input_mask': input_mask,
                'targets': family_label}
