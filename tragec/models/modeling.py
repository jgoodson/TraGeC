import logging
import typing

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torch.nn import LayerNorm

from .tape_model import TAPEModelMixin
from .configuration import BioConfig

logger = logging.getLogger(__name__)


class BioModel(pl.LightningModule, TAPEModelMixin):
    # From songlab-cal TAPE: https://github.com/songlab-cal/tape
    # Modified substantially
    r""" Base class for all models.

        :class:`~BioModel` takes care of storing the configuration of
        the models and handles methods for loading/downloading/saving models as well as a
        few methods commons to all models to (i) resize the sequence_rep embeddings and (ii) prune
        heads in the self-attention heads. These come from TAPEModelMixin and are derived from
        the methods the TAPE library uses for storing models.

        These require an _init_weights() method to be implemented by derived classes if they
        need initialization of anything not present in this version.

        BioModel also includes the setup to make thes complete Pytorch-Lightning modules.
        These methods include configure_optimizers, and the three step functions.

        These require a forward() and _compare() method to be implemented by derived classes.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~BioConfig`
              to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names`
              (string) as keys and `url` (string) of associated pretrained weights as values.
            - ``base_model_prefix``: a string indicating the attribute associated to the
              base model in derived classes of the same architecture adding modules on top
              of the base model.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BioConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class "
                "`BioConfig`. To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        # Save config in model
        self.config = config
        self.save_hyperparameters()

    def configure_optimizers(self) -> typing.Tuple[list, list]:
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
            from torch_optimizer import Lamb
            optimizer = Lamb(optimizer_grouped_parameters, lr=learning_rate)
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
                    raise ImportError(
                        "Specified optimizer {optimizer} is not available and torch_optimizer not available")
                OPT = getattr(torch_optimizer, optimizer, False)
                if OPT:
                    optimizer = OPT(optimizer_grouped_parameters, lr=learning_rate)
                else:
                    raise ImportError("Specified optimizer {optimizer} is not available")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.config.learning_rate,
                                                        total_steps=self.config.total_steps,
                                                        pct_start=self.config.warmup_steps / self.config.total_steps,
                                                        anneal_strategy='linear')

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def training_step(self, train_batch: typing.Dict, batch_idx: typing.Optional[int] = None) -> torch.Tensor:
        results = self.forward(**train_batch)

        loss, metrics = self._compare(results, train_batch)

        for k, m in metrics.items():
            self.log(f'train/{k}', m, sync_dist=True)

        self.log('train/loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch: typing.Dict, batch_idx: typing.Optional[int] = None) -> torch.Tensor:
        results = self.forward(**batch)

        loss, metrics = self._compare(results, batch)

        for k, m in metrics.items():
            self.log(f'val/{k}', m)

        self.log('val/loss', loss)

        return metrics

    def test_step(self, batch: typing.Dict, batch_idx: typing.Optional[int] = None):
        results = self.forward(**batch)

        loss, metrics = self._compare(results, batch)

        for k, m in metrics.items():
            self.log(f'test/{k}', m)

        self.log('test/loss', loss)


def create_sinusoidal_embeddings(n_pos, dim, out):
    out.requires_grad = False
    positions = torch.arange(0, n_pos)[:, None]
    dimensions = torch.arange(0, dim)
    position_enc = (positions / torch.pow(10000, 2 * (dimensions // 2) / dim)).to(out.device)
    out[:, 0::2] = torch.sin(position_enc[:, 0::2])
    out[:, 1::2] = torch.cos(position_enc[:, 1::2])


class ProteinEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.

    Modified From songlab-cal TAPE: https://github.com/songlab-cal/tape
    """

    def __init__(self, config: BioConfig, position_embeddings: bool = True):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        if position_embeddings:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size)
            if config.sinusoidal_pos_embds:
                create_sinusoidal_embeddings(
                    n_pos=config.max_position_embeddings, dim=config.hidden_size, out=self.position_embeddings.weight
                )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: typing.Optional[torch.Tensor] = None,
                position_ids: typing.Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        seq_length = input_ids.size()[1]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if hasattr(self, 'position_embeddings'):
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GeCEmbeddings(nn.Module):
    """Construct the embeddings from gene, (strand and spacing embeddings).
    """

    def __init__(self, config: BioConfig, position_embeddings: bool = True):
        super().__init__()
        self.generep_embeddings = nn.Linear(
            config.input_rep_size, config.hidden_size)
        if position_embeddings:
            self.position_embeddings: nn.Embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.sinusoidal_pos_embds:
                create_sinusoidal_embeddings(
                    n_pos=config.max_position_embeddings, dim=config.hidden_size, out=self.position_embeddings.weight
                )
        self.direction_embeddings: nn.Embedding = nn.Embedding(3, config.hidden_size)
        self.length_embeddings: nn.Embedding = nn.Embedding(config.gene_max_length // config.gene_length_bin_size + 1,
                                                            config.hidden_size)
        self.gene_length_bin_size = config.gene_length_bin_size
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                gene_reps: torch.Tensor,
                strands: typing.Optional[torch.Tensor] = None,
                lengths: typing.Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if strands is None:
            strands = torch.zeros_like(gene_reps[:, :, 0], dtype=torch.long)
        else:
            strands = strands.long()

        if lengths is None:
            lengths = torch.ones_like(gene_reps[:, :, 0], dtype=torch.long)
        else:
            lengths = strands.long()

        generep_embeddings = self.generep_embeddings(gene_reps)
        direction_embeddings = self.direction_embeddings(strands + 1)
        length_embeddings = self.length_embeddings(torch.clamp(lengths, 1, self.length_embeddings.num_embeddings) //
                                                   self.gene_length_bin_size)

        embeddings = generep_embeddings + direction_embeddings + length_embeddings
        if hasattr(self, 'position_embeddings'):
            position_ids = torch.arange(gene_reps.size()[1], dtype=torch.long, device=gene_reps.device)
            position_ids = position_ids.unsqueeze(0).expand(gene_reps.shape[:-1])
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


