# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modified by Roshan Rao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch T5 model. """

import logging

import torch
from torch import nn
from transformers.modeling_t5 import T5Config
from tape.models.modeling_utils import LayerNorm

from tragec.registry import registry
from .configuration import GeCConfig
from .modeling import GeCModel, GeCEmbeddings
from .modeling import MaskedReconHead
from .modeling_t5 import T5Stack

logger = logging.getLogger(__name__)

URL_PREFIX = "https://storage.googleapis.com/fire-tod.tryps.in/pytorch-models/"
T5_PRETRAINED_MODEL_ARCHIVE_MAP = {}
T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class GeCT5Config(T5Config, GeCConfig):
    pretrained_config_archive_map = T5_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 8096,
                 input_rep_size: int = 768,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 gradient_checkpointing: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_rep_size = input_rep_size
        self.d_model = hidden_size
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.n_positions = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
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


@registry.register_task_model('embed_gec', 'transformer5')
class GeCT5Model(GeCT5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embedding = GeCEmbeddings(config)

        self.t5 = T5Stack(config)

        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class GeCModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                gene_reps,
                input_mask=None):
        if input_mask is None:
            input_mask = torch.ones(gene_reps.shape[:-1])

        return self.t5(inputs_embeds=self.embedding(gene_reps))


@registry.register_task_model('masked_recon_modeling', 't5enc')
class GeCT5ForMaskedRecon(GeCT5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.t5 = GeCT5Model(config)

        self.projection = nn.Linear(
            config.hidden_size,
            config.input_rep_size,
        )
        self.mrm = MaskedReconHead(ignore=0)

        self.init_weights()

    def forward(self,
                gene_reps,
                input_mask=None,
                targets=None):
        outputs = self.t5(gene_reps, input_mask=input_mask)

        sequence_output = outputs[0]

        outputs = self.mrm(self.projection(sequence_output), targets)
        # (loss, ), prediction_scores, (hidden_states)
        return outputs
