import os
import logging
import copy
import typing
import json

import torch
import torch.optim as optim
import pytorch_lightning as pl
from ..utils.file_utils import cached_path
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from torch.nn import LayerNorm

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


class GeCConfig(object):
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
                 gene_length_bin_size: int = 32,
                 gene_max_length: int = 16384,
                 optimizer: str = 'adamw',
                 learning_rate: float = 1e-4,
                 warmup_steps: int = 1000,
                 total_steps: int = 0,
                 **kwargs):
        self.input_rep_size = input_rep_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.gene_length_bin_size = gene_length_bin_size
        self.gene_max_length = gene_max_length

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~GeCConfig.from_pretrained`
            class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the " \
                                              "model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a :class:`~GeCConfig`
             (or a derived class) from a pre-trained model configuration.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to
                  load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~GeCConfig.save_pretrained` method,
                  e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`,
                  e.g.: ``./my_model_directory/configuration.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict:
                key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will
                  be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration
                  attributes is controlled by the `return_unused_kwargs` keyword parameter.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)`
                  where `unused_kwargs` is a dictionary consisting of the key/value pairs
                  whose keys are not configuration attributes: ie the part of kwargs which
                  has not been used to update `config` and is otherwise ignored.

        Examples::

            # We can't instantiate directly the base class `GeCConfig` so let's
              show the examples on a derived class: ProteinBertConfig
            # Download configuration from S3 and cache.
            config = ProteinBertConfig.from_pretrained('bert-base-uncased')
            # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = ProteinBertConfig.from_pretrained('./test/saved_model/')
            config = ProteinBertConfig.from_pretrained(
                './test/saved_model/my_configuration.json')
            config = ProteinBertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True,
                foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
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
            return None
        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class GeCModel(pl.LightningModule):
    r""" Base class for all models.

        :class:`~GeCModel` takes care of storing the configuration of
        the models and handles methods for loading/downloading/saving models as well as a
        few methods commons to all models to (i) resize the input embeddings and (ii) prune
        heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~GeCConfig`
              to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names`
              (string) as keys and `url` (string) of associated pretrained weights as values.

            - ``base_model_prefix``: a string indicating the attribute associated to the
              base model in derived classes of the same architecture adding modules on top
              of the base model.
    """
    config_class: typing.Type[GeCConfig] = GeCConfig
    pretrained_model_archive_map: typing.Dict[str, str] = {}
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GeCConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class "
                "`GeCConfig`. To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        # Save config in model
        self.config = config

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if getattr(self.config, 'pruned_heads', False):
            self.prune_heads(self.config.pruned_heads)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and
                    associated values being the list of heads to prune in said layer
                    (list of `int`).
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~GeCModel.from_pretrained`
            ` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where " \
                                              "the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()``
        (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that
        the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used
        by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache
                  or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using
                  :func:`~GeCModel.save_pretrained`,
                  e.g.: ``./my_model_directory/``.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's
                ``__init__`` method

            config: (`optional`) instance of a class derived from
                :class:`~GeCConfig`: Configuration for the model to
                use instead of an automatically loaded configuation. Configuration can be
                automatically loaded when:

                - the model is a model provided by the library (loaded with the
                  ``shortcut-name`` string of a pretrained model), or
                - the model was saved using
                  :func:`~GeCModel.save_pretrained` and is reloaded
                  by suppling the save directory.
                - the model is loaded by suppling a local directory as
                  ``pretrained_model_name_or_path`` and a configuration JSON file named
                  `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state
                dictionary loaded from saved weights file. This option can be used if you
                want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~GeCModel.save_pretrained` and
                :func:`~GeCModel.from_pretrained` is not a
                simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override
                the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if
                such a file exists.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys,
                unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and
                initiate the model. (e.g. ``output_attention=True``). Behave differently
                depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwarg
                  directly passed to the underlying model's ``__init__`` method (we assume
                  all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the
                  configuration class initialization function
                  (:func:`~GeCConfig.from_pretrained`). Each key of
                  ``kwargs`` that corresponds to a configuration attribute will be used to
                  override said attribute with the supplied ``kwargs`` value. Remaining keys
                  that do not correspond to any configuration attribute will be passed to the
                  underlying model's ``__init__`` function.

        Examples::

            # Download model and configuration from S3 and cache.
            model = GeCBertModel.from_pretrained('bert-base-uncased')
            # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = GeCBertModel.from_pretrained('./test/saved_model/')
            # Update configuration during loading
            model = GeCBertModel.from_pretrained('bert-base-uncased', output_attention=True)
            assert model.config.output_attention == True

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        force_download = kwargs.pop("force_download", False)
        kwargs.pop("resume_download", False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir,
                                                force_download=force_download)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_model_archive_map.keys()),
                        archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        if cls.base_model_prefix not in (None, ''):
            if not hasattr(model, cls.base_model_prefix) and \
                    any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and \
                    not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs}
            return model, loading_info

        return model

    def configure_optimizers(self):
        learning_rate = self.config.learning_rate
        optimizer = self.config.optimizer
        param_optimizer = self.named_parameters()

        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if optimizer == 'adamw':
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer == 'lamb':
            from torch_optimizer import LAMB
            optimizer = LAMB(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
        elif optimizer == 'novograd':
            from torch_optimizer import NovoGrad
            optimizer = NovoGrad(optimizer_grouped_parameters, lr=learning_rate)
        elif isinstance(optimizer, str):
            OPT = getattr(optim, optimizer, False)
            if OPT:
                optimizer = OPT(optimizer_grouped_parameters, lr=learning_rate)
            else:
                try:
                    import torch_optimizer
                except ImportError:
                    raise NotImplemented(
                        "Specified optimizer {optimizer} is not available and torch_optimizer not available")
                OPT = getattr(torch_optimizer, optimizer, False)
                if OPT:
                    optimizer = OPT(optimizer_grouped_parameters, lr=learning_rate)
                else:
                    raise NotImplemented("Specified optimizer {optimizer} is not available")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.config.learning_rate,
                                                        total_steps=self.config.total_steps,
                                                        pct_start=self.config.warmup_steps / self.config.total_steps,
                                                        anneal_strategy='linear')

        return optimizer


class GeCEmbeddings(nn.Module):
    """Construct the embeddings from gene, (strand and spacing embeddings).
    """

    def __init__(self, config: GeCConfig, position_embeddings=True):
        super().__init__()
        self.generep_embeddings = nn.Linear(
            config.input_rep_size, config.hidden_size)
        if position_embeddings:
            self.position_embeddings: nn.Embedding = nn.Embedding(config.gene_max_length, config.hidden_size)
        self.direction_embeddings: nn.Embedding = nn.Embedding(3, config.hidden_size)
        self.length_embeddings: nn.Embedding = nn.Embedding(config.gene_max_length // config.gene_length_bin_size + 1,
                                                            config.hidden_size)
        self.gene_length_bin_size = config.gene_length_bin_size
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, gene_reps, strands=None, lengths=None):
        if strands is None:
            strands = torch.zeros_like(gene_reps[:, :, 0], dtype=torch.long)
        else:
            strands = strands.long()

        if lengths is None:
            lengths = torch.ones_like(gene_reps[:, :, 0], dtype=torch.long)
        else:
            lengths = strands.long()

        words_embeddings = self.generep_embeddings(gene_reps)
        direction_embeddings = self.direction_embeddings(strands + 1)
        length_embeddings = self.length_embeddings(torch.clamp(lengths, 1, self.length_embeddings.num_embeddings) //
                                                   self.gene_length_bin_size)

        embeddings = words_embeddings + direction_embeddings + length_embeddings
        if hasattr(self, 'position_embeddings'):
            position_ids = torch.arange(gene_reps.size(1), dtype=torch.long, device=gene_reps.device)
            position_ids = position_ids.unsqueeze(0).expand(gene_reps.shape[:-1])
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GeCDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 seqvec_type: str,
                 max_seq_len: int,
                 num_workers: int,
                 **opt_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.seqvec_type = seqvec_type
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        for kwarg, val in opt_kwargs.items():
            self.__setattr__(kwarg, val)


class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)


def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 512, num_labels)

    def forward(self, pooled_output):
        logits = self.classify(pooled_output)

        return logits
