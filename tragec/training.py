import typing
import logging
import json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from . import utils
from .registry import registry

logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]
OutputDict = typing.Dict[str, typing.Any]


def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              use_tpu: bool = True,
              no_cuda: bool = False,
              n_gpus: int = 1,
              max_grad_norm: float = 1.,
              seed: int = 42,
              num_workers: int = 8,
              patience: int = -1,
              resume_from_checkpoint: bool = False,
              eval_freq: int = 1,
              optimizer: str = 'adamw',
              max_seq_len: int = 512,
              percentmasked: float = .15,
              train_frac: float = 1.,
              val_frac: float = 1.,
              seqvec_type: str = 'seqvec') -> None:
    # SETUP AND LOGGING CODE #
    input_args = locals()

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir) / exp_dir

    # save all the hidden parameters.
    save_path.mkdir(parents=True, exist_ok=True)
    with (save_path / 'args.json').open('w') as f:
        json.dump(input_args, f)

    optional_dataset_args = {
        'percentmasked': percentmasked,
    }

    pl.seed_everything(seed)

    datamodule = registry.get_task_datamodule(task, data_dir, batch_size, seqvec_type, max_seq_len, num_workers,
                                              **optional_dataset_args)

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)

    model.config.optimizer = optimizer
    model.config.learning_rate = learning_rate
    model.config.warmup_steps = warmup_steps
    datamodule.setup()
    model.config.total_steps = len(datamodule.train_dataloader()) * num_train_epochs

    trainer_args = {
        'accumulate_grad_batches': gradient_accumulation_steps,
        'max_epochs': num_train_epochs,
        'log_every_n_steps': num_log_iter,
        'default_root_dir': save_path,
        'limit_train_batches': train_frac,
        'limit_val_batches': val_frac,
        'val_check_interval': eval_freq,
        'gradient_clip_val': max_grad_norm,
    }

    if patience != -1:
        es_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        trainer_args['callbacks'] = [es_callback]

    if fp16:
        trainer_args['precision'] = 16

    trainer_args['logger'] = pl_loggers.TensorBoardLogger(log_dir, name=exp_name)

    if no_cuda:
        trainer = pl.Trainer(gpus=n_gpus, **trainer_args)
    elif use_tpu:
        trainer = pl.Trainer(tpu_cores=8, **trainer_args)
    else:
        trainer = pl.Trainer(**trainer_args)

    trainer.fit(model=model, datamodule=datamodule)

    # TODO: Save model at end
    # TODO: implement checkpoint resume
    # TODO: Properly save hyperparameters


'''
def run_eval(model_type: str,
             task: str,
             from_pretrained: str,
             split: str = 'test',
             batch_size: int = 1024,
             model_config_file: typing.Optional[str] = None,
             data_dir: str = './data',
             no_cuda: bool = False,
             seed: int = 42,
             tokenizer: str = 'iupac',
             num_workers: int = 8,
             debug: bool = False,
             metrics: typing.Tuple[str, ...] = (),
             log_level: typing.Union[str, int] = logging.INFO,
             max_seq_len: int = 512,
             percentmasked: float = .15) -> typing.Dict[str, float]:
    local_rank = -1  # TAPE does not support torch.distributed.launch for evaluation
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    pretrained_dir = Path(from_pretrained)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}")

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)

    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()

    extra_args = {
        'max_seq_len': max_seq_len,
        'percentmasked': percentmasked,
    }
    valid_dataset = utils.setup_datamodule(task, data_dir, split, tokenizer, **extra_args)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        1, num_workers)

    metric_functions = [registry.get_metric(name) for name in metrics]
    save_outputs = run_eval_epoch(valid_loader, runner, is_master)
    target = [el['target'] for el in save_outputs]
    prediction = [el['prediction'] for el in save_outputs]

    metrics_to_save = {name: metric(target, prediction)
                       for name, metric in zip(metrics, metric_functions)}
    logger.info(''.join(f'{name}: {val}' for name, val in metrics_to_save.items()))

    with (pretrained_dir / 'results.pkl').open('wb') as f:
        pkl.dump((metrics_to_save, save_outputs), f)

    return metrics_to_save


def run_embed(model_type: str,
              data_file: str,
              out_file: str,
              from_pretrained: str,
              batch_size: int = 1024,
              model_config_file: typing.Optional[str] = None,
              full_sequence_embed: bool = False,
              no_cuda: bool = False,
              seed: int = 42,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              log_level: typing.Union[str, int] = logging.INFO) -> None:
    local_rank = -1  # TAPE does not support torch.distributed.launch for embedding
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}")

    task_spec = registry.get_task_spec('embed')
    model = registry.get_task_model(
        model_type, task_spec.name, model_config_file, from_pretrained)
    model = model.to(device)
    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()
    runner.eval()
    torch.set_grad_enabled(False)

    datamodule = task_spec.datamodule(data_file, tokenizer=tokenizer)  # type: ignore
    valid_loader = utils.setup_loader(datamodule, batch_size, local_rank, n_gpu, 1, num_workers)

    with utils.IncrementalNPZ(out_file) as npzfile:
        with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu):
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                outputs = runner.forward(batch, no_loss=True)
                ids = batch['ids']
                sequence_embed = outputs[0]
                pooled_embed = outputs[1]
                sequence_lengths = batch['input_mask'].sum(1)
                sequence_embed = sequence_embed.cpu().numpy()
                pooled_embed = pooled_embed.cpu().numpy()
                sequence_lengths = sequence_lengths.cpu().numpy()

                for seqembed, poolembed, length, protein_id in zip(
                        sequence_embed, pooled_embed, sequence_lengths, ids):
                    seqembed = seqembed[:length]
                    arrays = {'pooled': poolembed}
                    if not full_sequence_embed:
                        # avgpool across the sequence
                        arrays['avg'] = seqembed.mean(0)
                    else:
                        arrays['seq'] = seqembed
                    to_save = {protein_id: arrays}
                    npzfile.savez(**to_save)'''
