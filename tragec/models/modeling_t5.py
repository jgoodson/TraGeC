import torch
from torch import nn
import torch.utils.checkpoint

from transformers.modeling_t5 import T5PreTrainedModel, T5LayerNorm, \
    T5LayerSelfAttention, T5LayerFF

from .gradient_checkpointing import CheckpointFunction


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            position_bias=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
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
            output_hidden_states=None,
    ):

        output_attentions = self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prepare head mask if needed
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            features = (
                hidden_states,
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

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)
