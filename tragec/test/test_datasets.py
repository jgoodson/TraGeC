import unittest

from tragec.datasets import ProteinFluorescenceDataset, ProteinSecondaryStructureDataset, ProteinStabilityDataset, \
    ProteinRemoteHomologyDataset, ProteinnetDataset, ProteinDomainDataset, ProteinMaskedLanguageModelingDataset

from tragec.datasets import GeCMaskedReconstructionDataset, GeCClassificationDataset


class TestTAPEDataset(unittest.TestCase):
    config_cls = ProteinFluorescenceDataset
    item_type = tuple
    item_length = 3

    def setUp(self) -> None:
        self.dataset = self.config_cls(data_path='data', split='train')

    def test_length(self) -> None:
        assert len(self.dataset) == 2

    def test_getitem(self) -> None:
        assert isinstance(self.dataset[0], self.item_type)
        assert len(self.dataset[1]) == self.item_length

    def test_collate(self) -> None:
        batch = self.dataset.collate_fn([self.dataset[0], self.dataset[1]])
        assert (isinstance(batch, dict))


class TestSecondaryStructure(TestTAPEDataset):
    config_cls = ProteinSecondaryStructureDataset
    item_type = tuple
    item_length = 3


class TestStability(TestTAPEDataset):
    config_cls = ProteinStabilityDataset
    item_type = tuple
    item_length = 3


class TestHomology(TestTAPEDataset):
    config_cls = ProteinRemoteHomologyDataset
    item_type = tuple
    item_length = 3


class TestProteinnet(TestTAPEDataset):
    config_cls = ProteinnetDataset
    item_type = tuple
    item_length = 4


class TestDomain(TestTAPEDataset):
    config_cls = ProteinDomainDataset
    item_type = tuple
    item_length = 3


class TestMLM(TestTAPEDataset):
    config_cls = ProteinMaskedLanguageModelingDataset
    item_type = tuple
    item_length = 5


class TestTraGeCDataset(TestTAPEDataset):
    config_cls = GeCMaskedReconstructionDataset
    item_type = tuple
    item_length = 5

    def setUp(self) -> None:
        self.dataset = self.config_cls(data_path='data', split='train')

    def test_length(self) -> None:
        assert len(self.dataset) == 2

    def test_getitem(self) -> None:
        assert isinstance(self.dataset[0], self.item_type)
        assert len(self.dataset[1]) == self.item_length

    def test_collate(self) -> None:
        batch = self.dataset.collate_fn([self.dataset[0], self.dataset[1]])
        assert (isinstance(batch, dict))


class TestGeCClassification(TestTAPEDataset):
    config_cls = GeCClassificationDataset
    item_type = tuple
    item_length = 5


if __name__ == '__main__':
    unittest.main()
