import torch
from torch import nn

from tragec.registry import registry
from tragec.models.modeling import BioModel
from tragec.tasks.tasks import BioDataModule
from tragec.datasets import ProteinnetDataset


class BioContactPrediction(BioModel):
    # Adapted from songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, config):
        super().__init__(config)

        self._ignore_index = config.ignore_index

        self.classify = PairwiseContactPredictionHead(
            config.output_size, config.num_labels)

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        outputs = self.model(sequence_rep, input_mask=input_mask, **kwargs)

        sequence_output, pooled_output = outputs[:2]

        outputs = self.classify(sequence_output) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

    def _compare(self, results, batch):
        sequence_lengths = batch['protein_length']
        prediction = results[0]
        targets = batch['targets']
        loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
        contact_loss = loss_fct(prediction.view(-1, 2), targets.view(-1))
        metrics = {'precision_at_l5': self.classify.compute_precision_at_l5(sequence_lengths, prediction, targets)}
        return contact_loss, metrics


def create_contactpred_model(base_cls, base_model, name, seqtype):
    def __init__(self, config):
        base_cls.__init__(self, config)
        BioContactPrediction.__init__(self, config)
        self.model = base_model(config)
        self.tokenizer = 'iupac'
        self.init_weights()

    sc_model = type(
        f'{base_model.__name__.replace("Model", "")}ForContactPrediction',
        (base_cls, BioContactPrediction),
        {'__init__': __init__}
    )

    if seqtype == 'prot':
        registry.register_task_model('contact_prediction', f'{seqtype}_{name.lower()}', sc_model)

    return sc_model


@registry.register_task('contact_prediction')
class ProteinContactPredictionModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ProteinnetDataset
        self.split_names = ('train', 'valid', 'test')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'test'


class PairwiseContactPredictionHead(nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, input_size: int, ignore_index=-100):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(2 * input_size, 2))
        self._ignore_index = ignore_index

    def forward(self, inputs):
        prod = inputs[:, :, None, :] * inputs[:, None, :, :]
        diff = inputs[:, :, None, :] - inputs[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2

        # TODO: This is broken somehow. I'm not sure how it even works in TAPE because my test protein is
        # 405 AA long, the prediction is 405+cls/sep tokens + 1 pad token but the contact targets tensor is
        # 408 long.
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove start/stop tokens
        outputs = (prediction,)

        return outputs

    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = torch.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total
