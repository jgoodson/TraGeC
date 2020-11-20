"""PyTorch T5 model. """

import torch
from torch import nn
from transformers.modeling_t5 import T5Config

from tragec.registry import registry
from .modeling import GeCConfig, GeCModel, GeCEmbeddings
from .modeling_mrm import GeCMaskedRecon
from .modeling_singleclass import GeCSequenceClassification

import torch
from torch import nn
import torch.utils.checkpoint

from transformers.modeling_t5 import T5PreTrainedModel, T5LayerNorm, \
    T5LayerSelfAttention, T5LayerFF

URL_PREFIX = "https://storage.googleapis.com/fire-tod.tryps.in/pytorch-models/"
T5_PRETRAINED_MODEL_ARCHIVE_MAP = {}
T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            attention_mask,
            position_bias=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + attention_outputs
        return outputs  # (hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.checkpoint = config.checkpoint

        self.embed_tokens = embed_tokens

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            inputs_embeds=None,
            attention_mask=None,
    ):
        position_bias = None

        batch_size, seq_length = input_shape = inputs_embeds.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length).to(inputs_embeds.device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        extended_attention_mask.requires_grad = True

        hidden_states = self.dropout(inputs_embeds)
        for i, layer_module in enumerate(self.block):

            features = (
                hidden_states,
                extended_attention_mask,
                position_bias,
            )

            # If using gradient checkpointing, apply it layer-by-layer here
            if self.checkpoint:
                features = tuple(f for f in features if f is not None)
                parameters = dict(layer_module.named_parameters())
                layer_outputs = CheckpointFunction.apply(layer_module, len(features),
                                                         *(features + tuple(parameters.values())))
                # layer_outputs = torch.utils.checkpoint.checkpoint(layer_module, *features)
            else:
                layer_outputs = layer_module(*features)

            # layer_outputs is a tuple with:
            # hidden-states, (self-attention position bias)
            hidden_states = layer_outputs[0]

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention position bias)
                position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        outputs = (hidden_states,)

        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


class LayerNorm(nn.Module):  # type: ignore
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GeCT5Config(GeCConfig, T5Config):
    pretrained_config_archive_map = T5_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 max_position_embeddings: int = 8096,
                 gradient_checkpointing: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        T5Config.__init__(self, **kwargs)

        # Adapt comparable argument names from BertConfig for consistency
        self.d_model = hidden_size
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.n_positions = max_position_embeddings
        self.use_cache = False
        self.checkpoint = gradient_checkpointing


class GeCT5AbstractModel(GeCModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = GeCT5Config
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


# @registry.register_task_model('embed_gec', 'transformer5')
class GeCT5Model(GeCT5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embedding = GeCEmbeddings(config, position_embeddings=False)
        self.model = T5Stack(config)
        self.init_weights()

    def forward(self,
                gene_reps,
                input_mask=None,
                strands=None,
                lengths=None, ):
        return self.model(inputs_embeds=self.embedding(gene_reps, strands=strands, lengths=lengths),
                          attention_mask=input_mask)


@registry.register_task_model('masked_recon_modeling', 't5enc')
class GeCT5ForMaskedRecon(GeCT5AbstractModel, GeCMaskedRecon):

    def __init__(self, config):
        super().__init__(config)
        self.model = GeCT5Model(config)
        self.init_weights()


@registry.register_task_model('classify_gec', 't5enc')
class GeCBertForSequenceClassification(GeCT5AbstractModel, GeCSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.model = GeCT5Model(config)
        self.init_weights()
