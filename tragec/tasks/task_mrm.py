import numpy as np
import torch

from tragec.registry import registry
from tragec.models.modeling import BioModel
from .tasks import BioDataModule
from .task_mlm import PredictionHeadTransform
from tragec.datasets import GeCMaskedReconstructionDataset


class GeCMaskedRecon(BioModel):

    def __init__(self, config):
        super().__init__(config)

        hidden_act = 'gelu'
        layer_norm_eps = config.layer_norm_eps

        self.transform = PredictionHeadTransform(config.hidden_size, config.output_size, hidden_act, layer_norm_eps)

        self.decoder = torch.nn.Linear(
            config.hidden_size,
            config.input_rep_size,
        )

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                strands=None,
                lengths=None,
                **kwargs):
        outputs = self.model(sequence_rep, input_mask=input_mask, strands=strands, lengths=lengths)

        sequence_output = outputs[0]

        hidden_states = self.transform(sequence_output)
        hidden_states = self.decoder(hidden_states)

        # add hidden states and attention if they are here
        output = (hidden_states,) + outputs[2:]
        return output

    @staticmethod
    def _compare(results, batch):
        hidden_states = results[0]
        hidden_states = hidden_states.reshape((np.prod(hidden_states.shape[:2]),) + hidden_states.shape[2:])
        targets = batch['targets']
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
                'l1': torch.nn.L1Loss()(masked_states, masked_targets),
                'cosine': torch.nn.CosineEmbeddingLoss()(masked_states, masked_targets),
            }
            return masked_recon_loss, metrics


def create_mrm_model(base_cls, base_model, name, seqtype):
    def __init__(self, config):
        base_cls.__init__(self, config)
        GeCMaskedRecon.__init__(self, config)
        self.model = base_model(config)
        self.init_weights()

    mrm_model = type(
        f'{base_model.__name__.replace("Model", "")}ForMaskedRecon',
        (base_cls, GeCMaskedRecon),
        {'__init__': __init__}
    )

    if seqtype == 'gec':
        registry.register_task_model('masked_recon_modeling', f'gec_{name.lower()}', mrm_model)

    return mrm_model


@registry.register_task('masked_recon_modeling')
class GeCMaskedReconDataModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = GeCMaskedReconstructionDataset
        self.split_names = ('train', 'valid', 'holdout')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'holdout'
