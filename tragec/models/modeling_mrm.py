import torch
import numpy as np
from typing import Optional
from torch.utils.data import DataLoader, RandomSampler, Dataset

from .modeling import GeCModel, GeCDataModule
from ..datasets import MaskedReconstructionDataset
from ..utils._sampler import BucketBatchSampler
from tragec.registry import registry


class GeCMaskedRecon(GeCModel):

    def __init__(self, config):
        super().__init__(config)

        self.mrm_projection = torch.nn.Linear(
            config.hidden_size,
            config.input_rep_size,
        )

        self.init_weights()

    def forward(self,
                gene_reps,
                input_mask=None,
                strands=None,
                lengths=None,
                **kwargs):
        outputs = self.model(gene_reps, input_mask=input_mask, strands=strands, lengths=lengths)

        sequence_output = outputs[0]
        # add hidden states and attention if they are here

        outputs = self.mrm_projection(sequence_output)

        return outputs

    @staticmethod
    def _compare(hidden_states, targets):
        hidden_states = hidden_states.reshape((np.prod(hidden_states.shape[:2]),) + hidden_states.shape[2:])
        targets = targets.reshape((np.prod(targets.shape[:2]),) + targets.shape[2:])

        masked_states = hidden_states[targets.sum(1) != 0, :]
        masked_targets = targets[targets.sum(1) != 0, :]

        loss_fct = torch.nn.MSELoss()
        masked_recon_loss = loss_fct(
            masked_states, masked_targets)
        with torch.no_grad():
            double_t, double_s = masked_targets.type(torch.float32), masked_states.type(torch.float32)
            numerator = ((double_t - double_s) ** 2).sum(0)
            denominator = ((double_t - double_t.mean(0)) ** 2).sum(0)
            nonzero_denominator = denominator != 0
            nonzero_numerator = numerator != 0
            valid_score = nonzero_denominator & nonzero_numerator
            output_scores = torch.ones([double_t.shape[1]], dtype=torch.float32, device=numerator.device)
            output_scores[valid_score] = 1 - (numerator[valid_score] /
                                              denominator[valid_score])
            metrics = {
                'nse': output_scores.mean(),
            }
            return masked_recon_loss, metrics

    def training_step(self, train_batch, batch_idx):
        hidden_states = self.forward(**train_batch)
        targets = train_batch['targets']

        masked_recon_loss, metrics = self._compare(hidden_states, targets)

        for k, m in metrics.items():
            self.log(f'train_{k}', m, sync_dist=True)

        self.log('train_loss', masked_recon_loss, sync_dist=True)

        return masked_recon_loss

    def validation_step(self, batch, batch_idx):
        hidden_states = self.forward(**batch)
        targets = batch['targets']

        masked_recon_loss, metrics = self._compare(hidden_states, targets)

        for k, m in metrics.items():
            self.log(f'val_{k}', m)

        self.log('val_loss', masked_recon_loss)


@registry.register_task('masked_recon_modeling')
class GeCMaskedReconData(GeCDataModule):

    def setup(self, stage: Optional[str] = None) -> None:
        self.mrm_train = MaskedReconstructionDataset(
            data_path=self.data_dir,
            split='train',
            in_memory=False,
            seqvec_type=self.seqvec_type,
            max_seq_len=self.max_seq_len,
            percentmasked=self.percentmasked,
        )
        self.mrm_val = MaskedReconstructionDataset(
            data_path=self.data_dir,
            split='valid',
            in_memory=False,
            seqvec_type=self.seqvec_type,
            max_seq_len=self.max_seq_len,
            percentmasked=self.percentmasked,
        )
        self.mrm_test = MaskedReconstructionDataset(
            data_path=self.data_dir,
            split='holdout',
            in_memory=False,
            seqvec_type=self.seqvec_type,
            max_seq_len=self.max_seq_len,
            percentmasked=self.percentmasked,
        )

    def _prep_loader(self, dataset: MaskedReconstructionDataset) -> DataLoader:
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
        return self._prep_loader(self.mrm_train)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._prep_loader(self.mrm_val)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._prep_loader(self.mrm_test)
