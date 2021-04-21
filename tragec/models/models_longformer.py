"""PyTorch BERT model. """

import torch
from torch import nn
from transformers import LongformerModel, LongformerConfig

from .modeling import BioModel, GeCEmbeddings, ProteinEmbeddings, LayerNorm
from .configuration import BioConfig
from ..tasks.registry import create_and_register_models

URL_PREFIX = "https://models.fire.tryps.in/models/tragec/"
LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP = {}
LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class BioLongformerConfig(BioConfig, LongformerConfig):
    pretrained_config_archive_map = LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 attention_window: int = 64,
                 **kwargs):
        LongformerConfig.__init__(self,
                                  attention_window=attention_window,
                                  **kwargs)
        super().__init__(**kwargs)

        self.return_dict = True


class BioLongformerAbstractModel(BioModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    base_model_prefix = "longformer"
    config_class = BioLongformerConfig
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP


class BioLongformerModel(BioLongformerAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = LongformerModel(config)

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        global_attention_mask = torch.zeros(sequence_rep.shape[:2], device=sequence_rep.device)
        global_attention_mask[:, 0] = 1
        output = self.model(inputs_embeds=self.embedding(sequence_rep, **kwargs),
                            attention_mask=input_mask,
                            global_attention_mask=global_attention_mask)
        return output['last_hidden_state'], output['pooler_output']


class GeCLongformerModel(BioLongformerModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = GeCEmbeddings(config)


class ProteinLongformerModel(BioLongformerModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = ProteinEmbeddings(config)


create_and_register_models(locals(), BioLongformerAbstractModel, GeCLongformerModel, ProteinLongformerModel,
                           'longformer')
