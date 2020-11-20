import unittest

import numpy as np
import torch

from tragec.datasets import MaskedReconstructionDataset
from tragec.models.modeling_gecbert import GeCBertModel, GeCBertConfig, GeCBertForMaskedRecon

# from tragec.models.modeling_gect5 import GeCT5Config, GeCT5Model, GeCT5ForMaskedRecon

test_config_kwargs = dict(
    hidden_size=128,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=512,
    input_rep_size=128,
)


# Testing TODO:
# TODO: Test construction from dict/json

class TestGeCBertRaw(unittest.TestCase):

    def setUp(self) -> None:
        self.config = GeCBertConfig(**test_config_kwargs)
        self.model = GeCBertModel(self.config)

    def simpleForwardZeros(self, shape, strands: bool = None, lengths: int = None):
        if strands:
            strands = torch.ones(shape[:-1], dtype=torch.long)
        if lengths:
            lengths = torch.ones(shape[:-1], dtype=torch.long) * lengths

        (seq_output, pooled_output) = self.model(torch.zeros(shape, dtype=torch.float32),
                                                 strands=strands,
                                                 lengths=lengths)
        self.assertEqual(seq_output.shape, shape)
        self.assertEqual(pooled_output.shape, (shape[0], shape[2]))

    def test_forward(self) -> None:
        self.simpleForwardZeros((1, 100, 128))

    def test_forward_batch(self) -> None:
        self.simpleForwardZeros((4, 100, 128))

    def test_forward_strands(self) -> None:
        self.simpleForwardZeros((1, 100, 128), strands=True)

    def test_forward_lengths(self) -> None:
        self.simpleForwardZeros((1, 100, 128), lengths=100)

    def test_forward_lengths_over(self) -> None:
        self.simpleForwardZeros((1, 100, 128), lengths=100000)


class TestGeCBertRecon(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.config = GeCBertConfig(**test_config_kwargs)
        self.model = GeCBertForMaskedRecon(self.config)
        self.size = (1, 100, 128)

    def test_forward(self) -> None:
        target = torch.ones(self.size)
        seq_output = self.model(torch.zeros(self.size), targets=target)
        self.assertEqual(seq_output.shape, self.size)

    def test_backward(self) -> None:
        data = np.random.standard_normal(self.size[1:])
        data = data.astype(np.float32)
        m, t = MaskedReconstructionDataset._apply_pseudobert_mask(data)
        batch = MaskedReconstructionDataset.collate_fn(
            [(m, np.ones(len(m)), t, np.ones(self.size[1]), np.ones(self.size[1]) * 100)]
        )
        loss = self.model.training_step(batch, None)
        loss.backward()


'''
class TestGeCT5Raw(unittest.TestCase):

    def setUp(self) -> None:
        self.config = GeCT5Config(**test_config_kwargs)
        self.model = GeCT5Model(self.config)

    def simpleForwardZeros(self, shape, strands=None, lengths=None):
        (seq_output,) = self.model(torch.zeros(shape, dtype=torch.float32), strands=strands, lengths=lengths)
        self.assertEqual(seq_output.shape, shape)

    def test_forward(self) -> None:
        self.simpleForwardZeros((1, 100, 128))

    def test_forward_batch(self) -> None:
        self.simpleForwardZeros((4, 100, 128))

    def test_forward_strands(self) -> None:
        self.simpleForwardZeros((1, 100, 128), strands=torch.ones((1, 100), dtype=torch.long))
        self.simpleForwardZeros((1, 100, 128), strands=torch.zeros((1, 100), dtype=torch.long))
        self.simpleForwardZeros((1, 100, 128), strands=torch.zeros((1, 100), dtype=torch.long) - 1)

    def test_forward_lengths(self) -> None:
        self.simpleForwardZeros((1, 100, 128), lengths=torch.ones((1, 100), dtype=torch.long) * 100)

    def test_forward_lengths_over(self) -> None:
        self.simpleForwardZeros((1, 100, 128), lengths=torch.ones((1, 100), dtype=torch.long) * 100000)


class TestGeCT5RawCP(TestGeCT5Raw):

    def setUp(self) -> None:
        self.config = GeCT5Config(gradient_checkpointing=True, **test_config_kwargs)
        self.model = GeCT5Model(self.config)


class TestGeCT5Recon(TestGeCBertRecon):

    def setUp(self) -> None:
        super().setUp()
        self.config = GeCT5Config(**test_config_kwargs)
        self.model = GeCT5ForMaskedRecon(self.config)
        self.size = (1, 100, 128)

    def simpleForwardZeros(self, shape, ):
        (seq_output,) = self.model(torch.zeros(shape, dtype=torch.float32))
        self.assertEqual(seq_output.shape, shape)


class TestGeCT5ReconCP(TestGeCT5Recon):

    def setUp(self) -> None:
        super().setUp()
        self.config = GeCT5Config(gradient_checkpointing=True, **test_config_kwargs)
        self.model = GeCT5ForMaskedRecon(self.config)
        self.size = (1, 100, 128)
'''

if __name__ == '__main__':
    unittest.main()
