import os
import typing
import logging

import torch
from torch import nn
from transformers.file_utils import cached_path

from .configuration import BioConfig

logger = logging.getLogger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"

PathType = typing.Union[str, bytes, os.PathLike],


class TAPEModelMixin(nn.Module):
    config_class: typing.Type[BioConfig] = BioConfig
    pretrained_model_archive_map: typing.Dict[str, str] = {}
    base_model_prefix = ""

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights recursively on all children
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights

        If you use or need initialization for any other module types you should replace this method
        in your derived subclass.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_pretrained(self, save_directory: PathType):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~BioModel.from_pretrained`
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
    def _get_model(cls, pretrained_model_name_or_path):
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            archive_file = pretrained_model_name_or_path
        return archive_file

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: PathType,
                        config: typing.Optional[BioConfig] = None,
                        state_dict: typing.Optional[typing.Dict] = None,
                        cache_dir: PathType = None,
                        force_download: bool = False,
                        **model_kwargs):
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
                  :func:`~BioModel.save_pretrained`,
                  e.g.: ``./my_model_directory/``.

            config: (`optional`) instance of a class derived from
                :class:`~BioConfig`: Configuration for the model to
                use instead of an automatically loaded configuation. Configuration can be
                automatically loaded when:

                - the model is a model provided by the library (loaded with the
                  ``shortcut-name`` string of a pretrained model), or
                - the model was saved using
                  :func:`~BioModel.save_pretrained` and is reloaded
                  by suppling the save directory.
                - the model is loaded by suppling a local directory as
                  ``pretrained_model_name_or_path`` and a configuration JSON file named
                  `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state
                dictionary loaded from saved weights file. This option can be used if you
                want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~BioModel.save_pretrained` and
                :func:`~BioModel.from_pretrained` is not a
                simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override
                the cached versions if they exists.


            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and
                initiate the model. (e.g. ``output_attention=True``). Behave differently
                depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwarg
                  directly passed to the underlying model's ``__init__`` method (we assume
                  all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the
                  configuration class initialization function
                  (:func:`~BioConfig.from_pretrained`). Each key of
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
        # Load config
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir
            )

        # Load model
        archive_file = cls._get_model(pretrained_model_name_or_path)

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
            raise
        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))

        # Instantiate model.
        model = cls(config)

        if state_dict is None:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

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

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model
