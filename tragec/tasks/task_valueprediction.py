import torch

from tragec.registry import registry
from tragec.models.modeling import BioModel
from tragec.tasks.tasks import SimpleMLP, BioDataModule
from tragec.datasets import ProteinFluorescenceDataset, ProteinStabilityDataset


class ValuePredictionHead(torch.nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, hidden_size: int, dropout: float = 0.):
        super().__init__()
        self.value_prediction = SimpleMLP(hidden_size, 512, 1, dropout)

    def forward(self, pooled_output, targets=None):
        value_pred = self.value_prediction(pooled_output)
        outputs = (value_pred,)

        if targets is not None:
            loss_fct = torch.nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            outputs = (value_pred_loss,) + outputs
        return outputs  # (loss), value_prediction


class BioSequenceValuePrediction(BioModel):
    # Adapted from songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, config):
        super().__init__(config)

        self.value_prediction = ValuePredictionHead(
            config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        outputs = self.model(sequence_rep, input_mask=input_mask, **kwargs)

        sequence_output, pooled_output = outputs[:2]

        outputs = (self.value_prediction(pooled_output),) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

    def _compare(self, results, batch):
        value_pred = results[0]
        targets = batch['targets']
        loss_fct = torch.nn.MSELoss()
        classification_loss = loss_fct(value_pred, targets)
        metrics = {}
        return classification_loss, metrics


def create_valuepred_model(base_cls, base_model, name, seqtype):
    def __init__(self, config):
        base_cls.__init__(self, config)
        BioSequenceValuePrediction.__init__(self, config)
        self.model = base_model(config)
        self.tokenizer = 'iupac'
        self.init_weights()

    sc_model = type(
        f'{base_model.__name__.replace("Model", "")}ForValuePrediction',
        (base_cls, BioSequenceValuePrediction),
        {'__init__': __init__}
    )

    if seqtype == 'prot':
        registry.register_task_model('fluorescence', f'{seqtype}_{name.lower()}', sc_model)
        registry.register_task_model('stability', f'{seqtype}_{name.lower()}', sc_model)

    return sc_model


@registry.register_task('fluorescence')
class ProteinFluorescenceModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ProteinFluorescenceDataset
        self.split_names = ('train', 'valid', 'test')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'test'


@registry.register_task('stability')
class ProteinStabilityModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ProteinStabilityDataset
        self.split_names = ('train', 'valid', 'test')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'test'
