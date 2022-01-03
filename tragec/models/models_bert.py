"""PyTorch BERT model. """

from torch import nn
from transformers import BertModel, BertConfig

from .modeling import BioModel, GeCEmbeddings, ProteinEmbeddings, LayerNorm
from .configuration import BioConfig
from ..tasks.registry import create_and_register_models

URL_PREFIX = "https://models.fire.tryps.in/models/tragec/"
TAPE_URL_PREFIX = "https://s3.amazonaws.com/songlabdata/proteindata/pytorch-models/"
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {'prot-tiny_bert': URL_PREFIX + 'prot-tiny_bert-pytorch_model.bin',
                                     'bert-base': TAPE_URL_PREFIX + "bert-base-pytorch_model.bin",}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {'prot-tiny_bert': URL_PREFIX + 'prot-tiny_bert-config.json',
                                      'bert-base': TAPE_URL_PREFIX + "bert-base-config.json",}


class BioBertConfig(BioConfig, BertConfig):
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 **kwargs):
        BertConfig.__init__(self, **kwargs)
        super().__init__(**kwargs)
        self.return_dict = True


class BioBertAbstractModel(BioModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    base_model_prefix = "bert"
    config_class = BioBertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP

class BioBertModel(BioBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = BertModel(config)
        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        output = self.model(inputs_embeds=self.embeddings(sequence_rep, **kwargs),
                            attention_mask=input_mask)
        return output['last_hidden_state'], output['pooler_output']


class GeCBertModel(BioBertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = GeCEmbeddings(config)


class ProteinBertModel(BioBertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ProteinEmbeddings(config)

    @classmethod
    def _rewrite_module_name(cls, key: str) -> str:
        """
        Function to re-write module names when loading pretrained files
        """
        parts = key.split('.')
        if parts[1] in ('encoder', 'pooler'):
            parts.insert(1, 'model')
        return '.'.join(parts)



create_and_register_models(locals(), BioBertAbstractModel, GeCBertModel, ProteinBertModel, 'bert')
