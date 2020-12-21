import unittest

import random
import numpy as np
import torch

from tragec.datasets import GeCMaskedReconstructionDataset, ProteinMaskedLanguageModelingDataset
from tragec.models.models_bert import GeCBertModel, ProteinBertModel, BioBertConfig, GeCBertForMaskedRecon, \
    ProteinBertForMLM
from tragec.models.models_t5 import BioT5Config, GeCT5Model, GeCT5ForMaskedRecon

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
        self.config = BioBertConfig(**test_config_kwargs)
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
        self.config = BioBertConfig(**test_config_kwargs)
        self.model = GeCBertForMaskedRecon(self.config)
        self.size = (1, 100, 128)

    def test_forward(self) -> None:
        target = torch.ones(self.size)
        seq_output = self.model(torch.zeros(self.size), targets=target)[0]
        self.assertEqual(seq_output.shape, self.size)

    def test_backward(self) -> None:
        data = np.random.standard_normal(self.size[1:])
        data = data.astype(np.float32)
        m, t = GeCMaskedReconstructionDataset._apply_pseudobert_mask(data)
        batch = GeCMaskedReconstructionDataset.collate_fn(
            [(m, np.ones(len(m)), t, np.ones(self.size[1]), np.ones(self.size[1]) * 100)]
        )
        loss = self.model.training_step(batch, None)
        loss.backward()



class TestGeCT5Raw(unittest.TestCase):

    def setUp(self) -> None:
        self.config = BioT5Config(**test_config_kwargs)
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


class TestGeCBertRawCP(TestGeCBertRaw):

    def setUp(self) -> None:
        self.config = BioBertConfig(gradient_checkpointing=True, **test_config_kwargs)
        self.model = GeCBertModel(self.config)


class TestGeCT5RawCP(TestGeCT5Raw):

    def setUp(self) -> None:
        self.config = BioT5Config(gradient_checkpointing=True, **test_config_kwargs)
        self.model = GeCT5Model(self.config)


class TestGeCT5Recon(TestGeCBertRecon):

    def setUp(self) -> None:
        super().setUp()
        self.config = BioT5Config(**test_config_kwargs)
        self.model = GeCT5ForMaskedRecon(self.config)
        self.size = (1, 100, 128)

    def simpleForwardZeros(self, shape, ):
        (seq_output,) = self.model(torch.zeros(shape, dtype=torch.float32))
        self.assertEqual(seq_output.shape, shape)


class TestGeCT5ReconCP(TestGeCT5Recon):

    def setUp(self) -> None:
        super().setUp()
        self.config = BioT5Config(gradient_checkpointing=True, **test_config_kwargs)
        self.model = GeCT5ForMaskedRecon(self.config)
        self.size = (1, 100, 128)


class TestProtBertRaw(unittest.TestCase):

    def setUp(self) -> None:
        self.config = BioBertConfig(gradient_checkpointing=True, **test_config_kwargs)
        self.model = ProteinBertModel(self.config)

    def simpleForwardRandom(self, shape):
        (seq_output, pooled_output) = self.model(torch.from_numpy(np.random.randint(0, 30, shape)).long())
        self.assertEqual(seq_output.shape, shape + (self.config.hidden_size,))
        self.assertEqual(pooled_output.shape, (shape[0], self.config.hidden_size))

    def test_forward(self) -> None:
        self.simpleForwardRandom((1, 100))

    def test_forward_batch(self) -> None:
        self.simpleForwardRandom((4, 100))


class TestProtBertMLM(unittest.TestCase):
    tokens = ("<pad>",
              "<mask>",
              "<cls>",
              "<sep>",
              "<unk>",
              "A",
              "B",
              "C",
              "D",
              "E",
              "F",
              "G",
              "H",
              "I",
              "K",
              "L",
              "M",
              "N",
              "O",
              "P",
              "Q",
              "R",
              "S",
              "T",
              "U",
              "V",
              "W",
              "X",
              "Y",
              "Z",)

    def setUp(self) -> None:
        self.config = BioBertConfig(gradient_checkpointing=True, vocab_size=30, **test_config_kwargs)
        self.model = ProteinBertForMLM(self.config)
        self.size = (2, 100)

    def test_forward(self) -> None:
        input_tokens = target = torch.from_numpy(np.random.randint(0, 30, self.size)).long()
        seq_output = self.model(input_tokens, targets=target)[0]
        self.assertEqual(seq_output.shape, self.size + (self.config.vocab_size,))

    def test_backward(self) -> None:
        data = random.choices(self.tokens, k=self.size[1])
        ds = ProteinMaskedLanguageModelingDataset(None, 'train')
        masked_tokens, labels = ds._apply_bert_mask(data)
        masked_token_ids = np.array(
            ds.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            ds.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        batch = ds.collate_fn(
            [(masked_token_ids, input_mask, labels, None, None), ] * self.size[0]
        )
        loss = self.model.training_step(batch, None)
        loss.backward()


if __name__ == '__main__':
    unittest.main()
