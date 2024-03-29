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
    def _compare(results, batch, eps=1e-7):
        hidden_states = results[0]
        hidden_states = hidden_states.reshape((np.prod(hidden_states.shape[:2]),) + hidden_states.shape[2:])
        targets = batch['targets']
        targets = targets.reshape((np.prod(targets.shape[:2]),) + targets.shape[2:])

        masked_states = hidden_states[targets.sum(1) != 0, :]
        masked_targets = targets[targets.sum(1) != 0, :]

        loss_fct = torch.nn.MSELoss()
        loss_fct2 = torch.nn.CosineEmbeddingLoss()
        mse_loss = loss_fct(
            masked_states, masked_targets
        )
        cos_loss = loss_fct2(
            masked_states.view(masked_states.shape[0], -1),
            masked_targets.view(masked_states.shape[0], -1),
            torch.ones(masked_states.shape[0], device=masked_states.device)
        )

        numerator = ((masked_targets - masked_states) ** 2).sum(0) + eps
        denominator = ((masked_targets - masked_targets.mean(0)) ** 2).sum(0) + eps
        nse = (1 - (numerator / denominator)).mean()

        with torch.no_grad():
            metrics = {
                'nse': nse,
                'L1': torch.nn.L1Loss()(masked_states, masked_targets),
                'smL1': torch.nn.SmoothL1Loss()(masked_states, masked_targets),
                'mse': mse_loss,
                'cosine': cos_loss,
            }

        return 1 - nse, metrics


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
        registry.register_task_model('masked_recon_modeling_fp16', f'gec_{name.lower()}', mrm_model)

    return mrm_model


@registry.register_task('masked_recon_modeling')
@registry.register_task('masked_recon_modeling_fp16',
                        dataset_kwargs={'dtype': np.float16})
class GeCMaskedReconDataModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = GeCMaskedReconstructionDataset
        self.split_names = ('train', 'valid', 'holdout')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'holdout'
