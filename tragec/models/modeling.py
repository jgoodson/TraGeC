import logging
import typing

import numpy as np
import torch
from tape.models.modeling_utils import ProteinModel, ProteinConfig
from torch import nn

try:
    from apex.normalization import FusedLayerNorm as LayerNorm

    LayerNorm(1)
except (ImportError, ModuleNotFoundError):
    from torch.nn import LayerNorm

logger = logging.getLogger(__name__)


class GeCConfig(ProteinConfig):

    def __init__(self,
                 input_rep_size: int = 768,
                 layer_norm_eps: float = 1e-12,
                 hidden_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 gene_length_bin_size: int = 32,
                 gene_max_length: int = 16384,
                 **kwargs):
        super().__init__(**kwargs)
        try:
            self.hidden_size = 0
        except AttributeError:
            pass
        self.input_rep_size = input_rep_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.gene_length_bin_size = gene_length_bin_size
        self.gene_max_length = gene_max_length


class GeCModel(ProteinModel):
    r""" Base class for all models.
        :class:`~ProteinModel` takes care of storing the configuration of
        the models and handles methods for loading/downloading/saving models as well as a
        few methods commons to all models to (i) resize the input embeddings and (ii) prune
        heads in the self-attention heads.
        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~ProteinConfig`
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
        super().__init__(config, *inputs, **kwargs)


class GeCEmbeddings(nn.Module):
    """Construct the embeddings from gene, (strand and spacing embeddings).
    """

    def __init__(self, config: GeCConfig, position_embeddings=True):
        super().__init__()
        self.generep_embeddings = nn.Linear(
            config.input_rep_size, config.hidden_size)
        if position_embeddings:
            self.position_embeddings: nn.Embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
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


class MaskedReconHead(nn.Module):

    def __init__(self,
                 ignore: torch.tensor,  # ignored
                 loss_fn: torch.nn.Module = nn.MSELoss,
                 ):
        super().__init__()

        self._ignore = ignore
        self._loss_fn = loss_fn

    def forward(self, hidden_states, targets=None):
        outputs = (hidden_states,)
        hidden_states = hidden_states.reshape((np.prod(hidden_states.shape[:2]),) + hidden_states.shape[2:])
        if targets is not None:
            targets = targets.reshape((np.prod(targets.shape[:2]),) + targets.shape[2:])

            masked_states = hidden_states[targets.sum(1) != 0, :]
            masked_targets = targets[targets.sum(1) != 0, :]

            loss_fct = self._loss_fn()
            masked_recon_loss = loss_fct(
                masked_states, masked_targets)
            with torch.no_grad():
                double_t, double_s = masked_targets.type(torch.float32), masked_states.type(torch.float32)
                numerator = ((double_t - double_s) ** 2).sum(0)
                denominator = ((double_t - double_t.mean(0)) ** 2).sum(0)
                nonzero_denominator = denominator != 0
                nonzero_numerator = numerator != 0
                valid_score = nonzero_denominator & nonzero_numerator
                output_scores = torch.ones([double_t.shape[1]], dtype=torch.float32, device=numerator.device)
                output_scores[valid_score] = 1 - (numerator[valid_score] /
                                                  denominator[valid_score])
                metrics = {
                    'mean': torch.mean(double_s).detach().item(),
                    'mean_absval': torch.abs(double_s).mean().detach().item(),
                    'mean_stdev': torch.std(double_s, 0).mean().detach().item(),
                    'mean_nse': output_scores.mean().detach().item(),
                }

            for k, m in metrics.items():
                if np.isnan(m):
                    logger.warning(f'Metric {k} is nan ({m})')
                if np.isinf(m):
                    logger.warning(f'Metric {k} is inf ({m})')
            if torch.isnan(masked_recon_loss):
                logger.warning(f'Loss is nan ({masked_recon_loss})')
            if torch.isinf(masked_recon_loss):
                logger.warning(f'Loss is inf ({masked_recon_loss})')

            loss_and_metrics = (masked_recon_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs


class GeCMaskedRecon(GeCModel):

    def __init__(self, config):
        super().__init__(config)

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
