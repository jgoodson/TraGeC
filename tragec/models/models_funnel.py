"""PyTorch T5 model. """

import typing

from torch import nn
from torch.nn import LayerNorm
from transformers import FunnelConfig, FunnelModel

from .modeling import BioModel, GeCEmbeddings, ProteinEmbeddings
from .configuration import BioConfig
from ..tasks.registry import create_and_register_models

URL_PREFIX = "https://models.fire.tryps.in/models/tragec/"
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
        FunnelConfig.__init__(self, **kwargs)
        super().__init__(**kwargs)

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


class BioFunnelModel(BioFunnelAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = FunnelModel(config)
        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        # TODO: Implement encoder/decoder here with a config option to allow decoder-less use for some tasks
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
