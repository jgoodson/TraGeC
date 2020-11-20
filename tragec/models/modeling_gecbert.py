"""PyTorch BERT model. """

from torch import nn
from transformers import BertModel, BertConfig

from tragec.registry import registry
from .modeling import GeCConfig, GeCModel, GeCEmbeddings, LayerNorm
from .modeling_mrm import GeCMaskedRecon
from .modeling_singleclass import GeCSequenceClassification

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


# @registry.register_task_model('embed_gec', 'transformer')
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
class GeCBertForMaskedRecon(GeCBertAbstractModel, GeCMaskedRecon):

    def __init__(self, config):
        super().__init__(config)
        self.model = GeCBertModel(config)
        self.init_weights()


@registry.register_task_model('classify_gec', 'transformer')
class GeCBertForSequenceClassification(GeCBertAbstractModel, GeCSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.model = GeCBertModel(config)
        self.init_weights()
