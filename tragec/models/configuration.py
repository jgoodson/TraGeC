import copy
import json
import os
import logging

import typing
from transformers import cached_path

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"

PathType = typing.Union[str, bytes, os.PathLike],


class BioConfig(object):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    # Modified substantially
    """ Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods
        for loading/downloading/saving configurations.

        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` of with `short-cut-names`
                (string) as keys and `url` (string) of associated pretrained model
                configurations as values.

        Parameters:
            ``finetuning_task``: string, default `None`. Name of the task used to fine-tune
                the model.
            ``num_labels``: integer, default `2`. Number of classes to use when the model is
                a classification model (sequences/tokens)
            ``output_attentions``: boolean, default `False`. Should the model returns
                attentions weights.
            ``output_hidden_states``: string, default `False`. Should the model returns all
                hidden-states.
            ``torchscript``: string, default `False`. Is the model used with Torchscript.
    """
    pretrained_config_archive_map: typing.Dict[str, str] = {}

    def __init__(self,
                 input_rep_size: int = 768,
                 layer_norm_eps: float = 1e-12,
                 hidden_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 max_position_embeddings: int = 8192,
                 vocab_size: int = 30,
                 type_vocab_size: int = 2,
                 gene_length_bin_size: int = 32,
                 gene_max_length: int = 16384,
                 optimizer: str = 'adamw',
                 learning_rate: float = 1e-4,
                 warmup_steps: int = 1000,
                 total_steps: int = 0,
                 ignore_index: int = -1,
                 tokenizer: str = 'iupac',
                 gradient_checkpointing: bool = False,
                 sinusoidal_pos_embds: bool = False,
                 hidden_size: int = 768,
                 **kwargs):
        self.input_rep_size = input_rep_size
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.gene_length_bin_size = gene_length_bin_size
        self.gene_max_length = gene_max_length
        self.pruned_heads = None
        self.gradient_checkpointing = gradient_checkpointing
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.output_size = hidden_size

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def save_pretrained(self, save_directory: PathType) -> None:
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~BioConfig.from_pretrained`
            class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the " \
                                              "model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def _get_config(cls, pretrained_model_name_or_path):
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        return config_file

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: PathType,
                        cache_dir: PathType,
                        **kwargs):
        r""" Instantiate a :class:`~BioConfig`
             (or a derived class) from a pre-trained model configuration.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to
                  load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~BioConfig.save_pretrained` method,
                  e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`,
                  e.g.: ``./my_model_directory/configuration.json``.

            kwargs: (`optional`) dict:
                key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will
                  be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration
                  attributes is controlled by the `return_unused_kwargs` keyword parameter.

        Examples::

            # We can't instantiate directly the base class `BioConfig` so let's
              show the examples on a derived class: BioBertConfig
            # Download configuration from S3 and cache.
            config = BioBertConfig.from_pretrained('bert-base-uncased')
            # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BioBertConfig.from_pretrained('./test/saved_model/')
            config = BioBertConfig.from_pretrained(
                './test/saved_model/my_configuration.json')
            config = BioBertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True,
                foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        config_file = cls._get_config(pretrained_model_name_or_path)

        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained model "
                             "configuration file.".format(config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file))
            raise
        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        logger.info("Model config %s", config)
        return config

    @classmethod
    def from_dict(cls, json_object: dict):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file: PathType):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self) -> dict:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: PathType) -> None:
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())
