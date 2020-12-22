import torch
from torch import nn

from tragec.registry import registry
from tragec.models.modeling import BioModel
from tragec.tasks.tasks import accuracy, SimpleConv, BioDataModule
from tragec.datasets import ProteinSecondaryStructureDataset


class BioSequenceToSequenceClassification(BioModel):
    # Adapted from songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, config):
        super().__init__(config)

        self._ignore_index = config.ignore_index
        self.num_labels = config.num_labels

        self.classify = SequenceToSequenceClassificationHead(
            config.output_size, config.num_labels, ignore_index=self._ignore_index)

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        outputs = self.model(sequence_rep, input_mask=input_mask, **kwargs)

        sequence_output, pooled_output = outputs[:2]

        outputs = (self.classify(sequence_output),) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

    def _compare(self, results, batch):
        logits = results[0]
        targets = batch['targets']
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self._ignore_index)
        classification_loss = loss_fct(
            logits.view(-1, self.num_labels), targets.view(-1))
        acc_fct = Accuracy(ignore_index=self._ignore_index)
        metrics = {'accuracy':
                       acc_fct(logits.view(-1, self.num_labels), targets.view(-1))}
        return classification_loss, metrics


def create_seq2seqclass_model(base_cls, base_model, name, seqtype):
    def __init__(self, config):
        base_cls.__init__(self, config)
        BioSequenceToSequenceClassification.__init__(self, config)
        self.model = base_model(config)
        self.tokenizer = 'iupac'
        self.init_weights()

    sc_model = type(
        f'{base_model.__name__.replace("Model", "")}ForSeq2SeqClassification',
        (base_cls, BioSequenceToSequenceClassification),
        {'__init__': __init__}
    )

    if seqtype == 'prot':
        registry.register_task_model('secondary_structure', f'{seqtype}_{name.lower()}', sc_model)
    elif seqtype == 'gec':
        pass  # registry.register_task_model('secondary_structure', f'{seqtype}_{name.lower()}', sc_model)

    return sc_model


# TODO: GeC seq2seq per-gene classification dataset


@registry.register_task('secondary_structure', num_labels=3)
class ProteinSecondaryStructureModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ProteinSecondaryStructureDataset
        self.split_names = ('train', 'valid', 'casp12', 'ts115', 'cb513')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'casp12'


class Accuracy(nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)


class SequenceToSequenceClassificationHead(nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self,
                 input_size: int,
                 num_labels: int,
                 ignore_index: int = -100):
        super().__init__()
        self.classify = SimpleConv(
            input_size, 512, num_labels)
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = sequence_logits

        return outputs
