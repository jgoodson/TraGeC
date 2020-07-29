import logging
from typing import Type, Dict

from torch import nn
from tape.models.modeling_utils import LayerNorm

from transformers import PretrainedConfig, PreTrainedModel
from .modeling import GeCConfig, GeCModel, GeCEmbeddings, GeCMaskedRecon

logger = logging.getLogger(__name__)


def make_wrap_config(config_cls: Type[PretrainedConfig],
                     archive_map: Dict[str, str] = {},
                     config_key_remap: Dict[str, str] = {}):
    class NewConfig(config_cls, GeCConfig):
        pretrained_config_archive_map = archive_map

        def __init__(self,
                     input_rep_size: int = 512,
                     **kwargs):
            # Rename config arguments to maintain argument name consistency
            kwargs = {k if k not in config_key_remap else config_key_remap[k]: v for k, v in kwargs.items()}
            super().__init__(**kwargs)
            self.input_rep_size = input_rep_size

    return NewConfig


def make_wrap_absmodel(config_cls: Type[GeCConfig], archive_map: dict, model_prefix: str):
    class NewAbstractModel(GeCModel):
        """ An abstract class to handle weights initialization and
            a simple interface for dowloading and loading pretrained models.
        """
        config_class = config_cls
        pretrained_model_archive_map = archive_map
        base_model_prefix = model_prefix

        def _init_weights(self, module):
            """ Initialize the weights """
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    return NewAbstractModel


def make_wrap_basemodel(absmodel_cls: Type[GeCModel], basemodel_cls: Type[PreTrainedModel]):
    class NewModel(absmodel_cls):

        def __init__(self, config):
            super().__init__(config)

            self.embedding = GeCEmbeddings(config)

            self.model = basemodel_cls(config)

            self.init_weights()

        def forward(self,
                    gene_reps,
                    strands,
                    lengths):
            # TODO: attention mask?
            return self.model(inputs_embeds=self.embedding(gene_reps, strands=strands, lengths=lengths))

    return NewModel


def make_wrap_reconmodel(absmodel_cls: Type[GeCModel], model_cls: Type[GeCModel]):
    class NewForMaskedRecon(GeCMaskedRecon, absmodel_cls):

        def __init__(self, config):
            super().__init__(config)

            self.model = model_cls(config)

            self.init_weights()

    return NewForMaskedRecon


def wrap_model(config_cls: Type[PretrainedConfig],
               base_class: Type[PreTrainedModel],
               model_prefix: str,
               archive_map: dict = {}):
    wrapped_config_cls = make_wrap_config(config_cls, archive_map)
    new_abs_model = make_wrap_absmodel(wrapped_config_cls, archive_map, model_prefix)
    wrapped_base_model = make_wrap_basemodel(new_abs_model, base_class)
    wrapped_recon_model = make_wrap_reconmodel(new_abs_model, base_class)

    return wrapped_config_cls, wrapped_base_model, wrapped_recon_model
