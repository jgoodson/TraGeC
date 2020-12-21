import torch
from torch import nn

from tragec.registry import registry
from tragec.models.modeling import BioModel
from tragec.tasks.tasks import SimpleMLP, accuracy, BioDataModule
from tragec.datasets import GeCClassificationDataset, ProteinRemoteHomologyDataset


class BioSequenceClassification(BioModel):
    # Adapted from songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, config):
        super().__init__(config)

        self.classify = SequenceClassificationHead(
            config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        outputs = self.model(sequence_rep, input_mask=input_mask, **kwargs)

        sequence_output, pooled_output = outputs[:2]

        outputs = (self.classify(pooled_output),) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

    def _compare(self, results, batch):
        logits = results[0]
        targets = batch['targets']
        loss_fct = torch.nn.CrossEntropyLoss()
        classification_loss = loss_fct(logits, targets)
        metrics = {'mean_accuracy': accuracy(logits, targets)}
        return classification_loss, metrics


def create_seqclass_model(base_cls, base_model, name, seqtype):
    def __init__(self, config):
        base_cls.__init__(self, config)
        BioSequenceClassification.__init__(self, config)
        self.model = base_model(config)
        self.tokenizer = 'iupac'
        self.init_weights()

    sc_model = type(
        f'{base_model.__name__.replace("Model", "")}ForSequenceClassification',
        (base_cls, BioSequenceClassification),
        {'__init__': __init__}
    )

    if seqtype == 'prot':
        registry.register_task_model('remote_homology', f'{seqtype}_{name.lower()}', sc_model)
    elif seqtype == 'gec':
        registry.register_task_model('classify_gec', f'{seqtype}_{name.lower()}', sc_model)

    return sc_model


@registry.register_task('classify_gec', num_labels=19)
class GeCSequenceClassificationDataModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = GeCClassificationDataset
        self.split_names = ('train', 'valid')
        self.train_split = 'train'
        self.val_split = 'valid'


@registry.register_task('remote_homology', num_labels=1195)
class ProteinRemoteHomologyModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ProteinRemoteHomologyDataset
        self.split_names = ('train', 'valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'test_fold_holdout'


class SequenceClassificationHead(nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 512, num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        logits = self.classify(pooled_output)

        return logits
