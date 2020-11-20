import torch
from typing import Optional
from torch.utils.data import DataLoader, RandomSampler

from .modeling import GeCModel, GeCDataModule, SequenceClassificationHead, accuracy
from ..datasets import GeCClassificationDataset
from ..utils._sampler import BucketBatchSampler
from tragec.registry import registry


class GeCSequenceClassification(GeCModel):

    def __init__(self, config):
        super().__init__(config)

        self.classify = SequenceClassificationHead(
            config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self,
                gene_reps,
                input_mask=None,
                strands=None,
                lengths=None,
                **kwargs):
        outputs = self.model(gene_reps, input_mask=input_mask, strands=strands, lengths=lengths)

        sequence_output, pooled_output = outputs[:2]

        outputs = self.classify(pooled_output) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

    def _compare(self, batch):
        logits = self.forward(**batch)[0]
        targets = batch['targets']
        loss_fct = torch.nn.CrossEntropyLoss()
        classification_loss = loss_fct(logits, targets)
        metrics = {'mean_accuracy': accuracy(logits, targets)}
        return classification_loss, metrics

    def training_step(self, train_batch, batch_idx):

        classification_loss, metrics = self._compare(train_batch)

        for k, m in metrics.items():
            self.log(f'train_{k}', m)

        self.log('train_loss', classification_loss)

        return classification_loss

    def validation_step(self, batch, batch_idx):
        classification_loss, metrics = self._compare(batch)

        for k, m in metrics.items():
            self.log(f'val_{k}', m)

        self.log('val_loss', classification_loss)


@registry.register_task('classify_gec', num_labels=19)
class GeCSeqClassData(GeCDataModule):

    def setup(self, stage: Optional[str] = None) -> None:
        self.sc_train = GeCClassificationDataset(
            data_path=self.data_dir,
            split='train',
            in_memory=False,
            seqvec_type=self.seqvec_type,
            max_seq_len=self.max_seq_len,
        )
        self.sc_val = GeCClassificationDataset(
            data_path=self.data_dir,
            split='valid',
            in_memory=False,
            seqvec_type=self.seqvec_type,
            max_seq_len=self.max_seq_len,
        )

    def _prep_loader(self, dataset: GeCClassificationDataset) -> DataLoader:
        sampler = RandomSampler(dataset)

        batch_sampler = BucketBatchSampler(
            sampler, self.batch_size, False, dataset, 10)

        loader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            batch_sampler=batch_sampler,
        )
        return loader

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._prep_loader(self.sc_train)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._prep_loader(self.sc_val)
