import unittest
import tempfile

from tragec import BioConfig
from tragec.models.models_bert import GeCBertModel, BioBertConfig

test_config_kwargs = dict(
    hidden_size=128,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=512,
    input_rep_size=128,
)


class TestConfig(unittest.TestCase):
    config_cls = BioConfig

    def setUp(self) -> None:
        self.config = self.config_cls()

    def test_create_from_args(self) -> None:
        self.config = self.config_cls(**test_config_kwargs)

    def test_save(self) -> None:
        with tempfile.TemporaryDirectory() as savedir:
            self.config.save_pretrained(savedir)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as savedir:
            self.config.save_pretrained(savedir)
            self.config_cls.from_pretrained(savedir, cache_dir=None)


class TestBertConfig(TestConfig):
    config_cls = BioBertConfig

    def test_create_model(self) -> None:
        self.model = GeCBertModel(self.config)


if __name__ == '__main__':
    unittest.main()
