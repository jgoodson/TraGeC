"""PyTorch T5 model. """

from torch import nn
from torch.nn import LayerNorm
from transformers import T5Config

from .modeling import BioModel, GeCEmbeddings, ProteinEmbeddings
from .configuration import BioConfig
from ..tasks.registry import create_and_register_models
from .utils_t5 import T5Stack

URL_PREFIX = "https://models.fire.tryps.in/models/tragec/"
T5_PRETRAINED_MODEL_ARCHIVE_MAP = {}
T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class BioT5Config(BioConfig, T5Config):
    pretrained_config_archive_map = T5_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 **kwargs):
        T5Config.__init__(self, **kwargs)
        super().__init__(hidden_size=hidden_size,
                         **kwargs)

        # Adapt comparable argument names from T5Config for consistency with BioBertConfig
        self.d_model = hidden_size
        if 'intermediate_size' in kwargs:
            self.d_ff = kwargs['intermediate_size']
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.use_cache = False
        self.feed_forward_proj = 'gated-gelu'

    @property
    def hidden_size(self):
        return self.d_model

    @hidden_size.setter
    def hidden_size(self, value):
        self.d_model = 0


class BioT5AbstractModel(BioModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BioT5Config
    pretrained_model_archive_map = T5_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "t5"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BioT5Model(BioT5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = T5Stack(config)
        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        return self.model(inputs_embeds=self.embedding(sequence_rep, **kwargs),
                          attention_mask=input_mask)


class GeCT5Model(BioT5Model):

    def __init__(self, config):
        super().__init__(config)

        self.embedding = GeCEmbeddings(config, position_embeddings=False)


class ProteinT5Model(BioT5Model):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = ProteinEmbeddings(config, position_embeddings=False)


create_and_register_models(locals(), BioT5AbstractModel, GeCT5Model, ProteinT5Model, 't5')
