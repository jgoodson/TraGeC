"""Implementation of a bucketed data sampler from PyTorch-NLP.
Modified by Roshan Rao. (and then further)

# From songlab-cal TAPE: https://github.com/songlab-cal/tape

See https://github.com/PetrochukM/PyTorch-NLP/
"""
import math
import operator
import typing

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SubsetRandomSampler

from tragec.datasets import BioDataset


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.
    Args:
        dataset (iterable): Iterable data.
     Example:
        >>> list(SortedSampler(range(10)))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self,
                 dataset: BioDataset,
                 indices: typing.Optional[typing.Iterable[int]]):
        super().__init__(dataset)
        self.dataset = dataset

        if indices is None:
            sort_keys = ((i, dataset.item_length(i)) for i in range(len(dataset)))
        else:
            sort_keys = ((i, dataset.item_length(i)) for i in indices)
        self.sorted_indices = [i for i, _ in sorted(sort_keys, reverse=True, key=operator.itemgetter(1))]

    def __iter__(self):
        return iter(self.sorted_indices)

    def __len__(self):
        return len(self.dataset)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.
    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted
    and vice versa. Provides ~10-25 percent speedup.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular
        libraries like ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together
        examples with a similar size length to reduce the padding required for each batch
        while maintaining some noise through bucketing.

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size
            would be less than `batch_size`.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    Example:
        >>> import torch.utils.data.sampler
        >>> sampler = torch.utils.data.sampler.SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool,
                 dataset: BioDataset,
                 bucket_size_multiplier: typing.Union[int, float] = 100):
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = dataset
        self.bucket_sampler = BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)), False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(self.dataset, indices=bucket)
            yield from SubsetRandomSampler(
                list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))
            )

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)
