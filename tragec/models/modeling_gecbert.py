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
"""PyTorch BERT model. """

import logging

import torch

from tape.models.modeling_utils import LayerNorm
from torch import nn
from transformers import BertModel, BertConfig

from tragec.registry import registry
from .modeling import GeCConfig, GeCModel, GeCEmbeddings, GeCMaskedRecon, MaskedReconHead

logger = logging.getLogger(__name__)

URL_PREFIX = "https://storage.googleapis.com/fire-tod.tryps.in/pytorch-models/"
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class GeCBertConfig(GeCConfig, BertConfig):
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        BertConfig.__init__(self, **kwargs)


class GeCBertAbstractModel(GeCModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = GeCBertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_task_model('embed_gec', 'transformer')
class GeCBertModel(GeCBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embedding = GeCEmbeddings(config)

        self.model = BertModel(config)

        self.init_weights()

    def forward(self,
                gene_reps,
                input_mask=None,
                strands=None,
                lengths=None, ):
        return self.model(inputs_embeds=self.embedding(gene_reps, strands=strands, lengths=lengths),
                          attention_mask=input_mask)


@registry.register_task_model('masked_recon_modeling', 'transformer')
class GeCBertForMaskedRecon(GeCBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = GeCBertModel(config)

        self.projection = nn.Linear(
            config.hidden_size,
            config.input_rep_size,
        )
        self.mrm = MaskedReconHead(ignore=0)

        self.init_weights()

    def forward(self,
                gene_reps,
                targets=None,
                input_mask=None,
                strands=None,
                lengths=None):
        outputs = self.model(gene_reps, input_mask=input_mask, strands=strands, lengths=lengths)

        sequence_output = outputs[0]
        # add hidden states and attention if they are here

        outputs = self.mrm(self.projection(sequence_output), targets)

        return outputs
