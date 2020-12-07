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
from tape.models.modeling_utils import SequenceClassificationHead

from tragec.registry import registry
from .modeling import GeCConfig, GeCModel, GeCEmbeddings, GeCMaskedRecon
from .modeling_t5 import T5Stack

logger = logging.getLogger(__name__)

URL_PREFIX = "https://storage.googleapis.com/fire-tod.tryps.in/pytorch-models/"
T5_PRETRAINED_MODEL_ARCHIVE_MAP = {}
T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


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


@registry.register_task_model('embed_gec', 'transformer5')
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
class GeCT5ForMaskedRecon(GeCMaskedRecon, GeCT5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = GeCT5Model(config)

        self.init_weights()


@registry.register_task_model('classify_gec', 't5enc')
class GeCBertForSequenceClassification(GeCT5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = GeCT5Model(config)
        self.classify = SequenceClassificationHead(
            config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self,
                gene_reps,
                targets=None,
                input_mask=None,
                strands=None,
                lengths=None):
        outputs = self.model(gene_reps, input_mask=input_mask, strands=strands, lengths=lengths)

        sequence_output = outputs[0]

        outputs = self.classify(sequence_output.mean(1), targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
