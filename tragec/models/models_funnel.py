"""PyTorch T5 model. """

import typing

from torch import nn
from torch.nn import LayerNorm
from transformers import FunnelConfig, FunnelModel

from .modeling import BioConfig, BioModel, GeCEmbeddings, ProteinEmbeddings
from ..tasks.registry import create_and_register_models

URL_PREFIX = "http://macpro.tryps.in:8080/models/tragec/"
FUNNEL_PRETRAINED_MODEL_ARCHIVE_MAP = {}
FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class BioFunnelConfig(BioConfig, FunnelConfig):
    pretrained_config_archive_map = FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 hidden_size: int = 768,
                 block_sizes: typing.Optional[list] = None,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 **kwargs):
        super().__init__(**kwargs)
        FunnelConfig.__init__(self, **kwargs)

        # Adapt comparable argument names from FunnelConfig for consistency with BioBertConfig

        self.d_model = hidden_size
        if block_sizes:
            self.block_sizes = block_sizes
        else:
            self.block_sizes = [num_hidden_layers // 3] * 3
        self.n_head = num_attention_heads
        self.use_cache = False

    @property
    def hidden_size(self):
        return self.d_model

    @hidden_size.setter
    def hidden_size(self, value):
        self.d_model = 0


class BioFunnelAbstractModel(BioModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BioFunnelConfig
    pretrained_model_archive_map = FUNNEL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "funnel"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BioFunnelModel(BioFunnelAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = FunnelModel(config)
        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        return self.model(inputs_embeds=self.embedding(sequence_rep, **kwargs),
                          attention_mask=input_mask)


class GeCFunnelModel(BioFunnelModel):

    def __init__(self, config):
        super().__init__(config)

        self.embedding = GeCEmbeddings(config, position_embeddings=False)


class ProteinFunnelModel(BioFunnelModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = ProteinEmbeddings(config, position_embeddings=False)


create_and_register_models(locals(), BioFunnelAbstractModel, GeCFunnelModel, ProteinFunnelModel, 'funnel')
