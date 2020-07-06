import logging

from tape.models.modeling_utils import ProteinConfig

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

logger = logging.getLogger(__name__)


class GeCConfig(ProteinConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
