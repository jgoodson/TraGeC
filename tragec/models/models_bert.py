"""PyTorch BERT model. """

from torch import nn
from transformers import BertModel, BertConfig

from .modeling import BioConfig, BioModel, GeCEmbeddings, ProteinEmbeddings, LayerNorm
from ..tasks.registry import create_and_register_models

URL_PREFIX = "https://models.fire.tryps.in/models/tragec/"
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {'prot-tiny_bert': URL_PREFIX + 'prot-tiny_bert-pytorch_model.bin'}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {'prot-tiny_bert': URL_PREFIX + 'prot-tiny_bert-config.json'}


class BioBertConfig(BioConfig, BertConfig):
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        BertConfig.__init__(self, **kwargs)
        self.return_dict = True


class BioBertAbstractModel(BioModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    base_model_prefix = "bert"
    config_class = BioBertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BioBertModel(BioBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = BertModel(config)

        self.init_weights()

    def forward(self,
                sequence_rep,
                input_mask=None,
                **kwargs):
        output = self.model(inputs_embeds=self.embedding(sequence_rep, **kwargs),
                            attention_mask=input_mask)
        return output['last_hidden_state'], output['pooler_output']


class GeCBertModel(BioBertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = GeCEmbeddings(config)


class ProteinBertModel(BioBertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = ProteinEmbeddings(config)


create_and_register_models(locals(), BioBertAbstractModel, GeCBertModel, ProteinBertModel, 'bert')
