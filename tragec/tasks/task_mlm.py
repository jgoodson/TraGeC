import typing

import torch
from torch import nn
from torch.nn import LayerNorm

from tragec.registry import registry
from tragec.models.modeling import BioModel
from tragec import BioConfig
from tragec.tasks.tasks import get_activation_fn, BioDataModule
from tragec.datasets import ProteinMaskedLanguageModelingDataset


class ProteinMLM(BioModel):
    # Adapted from songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self, config: BioConfig):
        super().__init__(config)

        hidden_act = 'gelu'
        layer_norm_eps = config.layer_norm_eps
        ignore_index = config.ignore_index

        self.transform = PredictionHeadTransform(config.hidden_size,
                                                 config.output_size,
                                                 hidden_act,
                                                 layer_norm_eps)

        self.decoder = nn.Linear(config.output_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(config.vocab_size))  # type: ignore
        self.vocab_size = config.vocab_size
        self._ignore_index = ignore_index

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        outputs = self.model(sequence_rep, input_mask=input_mask, )

        sequence_output = outputs[0]

        hidden_states = self.transform(sequence_output)
        hidden_states = self.decoder(hidden_states) + self.bias
        output = (hidden_states,) + outputs[2:]
        return output

    def _compare(self, results, batch):
        hidden_states = results[0]
        targets = batch['targets']
        loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
        masked_lm_loss = loss_fct(
            hidden_states.view(-1, self.vocab_size), targets.view(-1))
        metrics = {'perplexity': torch.exp(masked_lm_loss)}
        return masked_lm_loss, metrics


def create_mlm_model(base_cls, base_model, name, seqtype):
    def __init__(self, config):
        base_cls.__init__(self, config)
        ProteinMLM.__init__(self, config)
        self.model = base_model(config)
        self.tokenizer = 'iupac'
        self.init_weights()

    mlm_model = type(
        f'{base_model.__name__.replace("Model", "")}ForMLM',
        (base_cls, ProteinMLM),
        {'__init__': __init__}
    )

    if seqtype == 'prot':
        registry.register_task_model('masked_language_modeling', f'prot_{name.lower()}', mlm_model)

    return mlm_model


# noinspection PyAbstractClass
@registry.register_task('masked_language_modeling')
class ProteinMLMDataModule(BioDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = ProteinMaskedLanguageModelingDataset
        self.split_names = ('train', 'valid', 'holdout')
        self.train_split = 'train'
        self.val_split = 'valid'
        self.test_split = 'holdout'


class PredictionHeadTransform(nn.Module):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    def __init__(self,
                 hidden_size: int,
                 input_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = get_activation_fn(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
