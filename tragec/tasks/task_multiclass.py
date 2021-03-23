import torch
from torch import nn

from tragec.registry import registry
from tragec.models.modeling import BioModel
from tragec.tasks.tasks import SimpleConv, BioDataModule
from tragec.datasets import ProteinDomainDataset
from tragec.tasks.extras_multiclass import pos_weights


class BioSequenceMultiClassification(BioModel):

    def __init__(self, config):
        super().__init__(config)

        self.classify = MultiLabelClassificationHead(
            config.output_size, config.num_labels)
        self.pos_weights = config.pos_weights
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
        targets = batch['targets']
        logits = results[0]

        if self.pos_weights:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weights, device=logits.device))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        classification_loss = loss_fct(logits, targets)

        # Roughly calculate best thresholds per-sequence based on F1-score
        thresholds, metrics = optimize_thresholds(logits.detach(), targets)

        f1, precision, recall, accuracy = metrics.mean(0)

        metrics = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'mean_prob': torch.sigmoid(logits).mean(),
        }
        loss_and_metrics = (classification_loss, metrics)
        return loss_and_metrics


def create_multiclass_model(base_cls, base_model, name, seqtype):
    def __init__(self, config):
        base_cls.__init__(self, config)
        BioSequenceMultiClassification.__init__(self, config)
        self.model = base_model(config)
        self.tokenizer = 'iupac'
        self.init_weights()

    mc_model = type(
        f'{base_model.__name__.replace("Model", "")}ForMultilabelClassification',
        (base_cls, BioSequenceMultiClassification),
        {'__init__': __init__}
    )

    if seqtype == 'prot':
        registry.register_task_model('protein_domain', f'{seqtype}_{name.lower()}', mc_model)

    return mc_model


@registry.register_task('protein_domain', model_kwargs={'num_labels': 14808, 'pos_weights': pos_weights})
class ProteinDomainPredictionModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ProteinDomainDataset
        self.split_names = ('train', 'valid', 'holdout')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'holdout'


class MultiLabelClassificationHead(nn.Module):
    def __init__(self, input_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleConv(input_size, 512, num_labels)

    def forward(self, sequence_output):
        logits = self.classify(sequence_output).max(1)[0]
        outputs = (logits,)

        return outputs  # (loss), logits


def scores(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    The original implmentation is written by Michal Haltuf on Kaggle.
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    '''

    assert y_true.ndim in (1, 2)
    assert y_pred.ndim in (1, 2)

    tp = (y_true * y_pred).sum(-1).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(-1).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(-1).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(-1).to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return torch.stack((f1, precision, recall, accuracy))


def optimize_thresholds(c: torch.Tensor, y: torch.Tensor):
    best_t = torch.zeros(c.shape[0])
    best_metrics = torch.zeros((c.shape[0], 4))
    probs = torch.sigmoid(c.float())
    for t in range(1, 100):
        t = t / 100
        metrics = scores(y, (probs > t).float())
        for i, (tscore, pre_m) in enumerate(zip(metrics.T, best_metrics)):
            if tscore[0] > pre_m[0]:
                best_metrics[i], best_t[i] = tscore, t
    return best_t, best_metrics
