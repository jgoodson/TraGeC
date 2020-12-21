import logging
import os
import pickle as pkl
import typing
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import lr_monitor

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
              checkpoint_file: typing.Optional[str] = None,
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
              seqvec_type: str = 'seqvec',
              ) -> None:
    # SETUP AND LOGGING CODE #
    # input_args = locals()

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir) / exp_dir

    # save all the hidden parameters.
    save_path.mkdir(parents=True, exist_ok=True)
    # with (save_path / 'args.json').open('w') as f:
    #    json.dump(input_args, f)

    optional_dataset_args = {
        'percentmasked': percentmasked,
    }

    if resume_from_checkpoint:
        if not checkpoint_file:
            checkpoint_file = [f for f in os.listdir(save_path) if f.endswith('.ckpt')]

        model = registry.get_task_model(model_type, task, checkpoint_file, model_config_file, from_pretrained)

        datamodule = registry.get_task_datamodule(task, data_dir, batch_size, max_seq_len, num_workers, seqvec_type,
                                                  model.config.tokenizer, **optional_dataset_args)

        trainer = pl.Trainer(resume_from_checkpoint=checkpoint_file)

    else:
        datamodule, model, trainer = prepare_trainer(batch_size, checkpoint_file, data_dir, eval_freq, exp_name, fp16,
                                                     from_pretrained, gradient_accumulation_steps, learning_rate,
                                                     log_dir, max_grad_norm, max_seq_len, model_config_file, model_type,
                                                     n_gpus, no_cuda, num_log_iter, num_train_epochs, num_workers,
                                                     optimizer, patience, optional_dataset_args, save_path, seed,
                                                     seqvec_type,
                                                     task, train_frac, use_tpu, val_frac, warmup_steps)

    trainer.fit(model=model, datamodule=datamodule)

    model.save_pretrained(save_path)


def prepare_trainer(batch_size, checkpoint_file, data_dir, eval_freq, exp_name, fp16, from_pretrained,
                    gradient_accumulation_steps, learning_rate, log_dir, max_grad_norm, max_seq_len, model_config_file,
                    model_type, n_gpus, no_cuda, num_log_iter, num_train_epochs, num_workers, optimizer, patience,
                    optional_dataset_args, save_path, seed, seqvec_type, task, train_frac, use_tpu, val_frac,
                    warmup_steps):
    pl.seed_everything(seed)
    model = registry.get_task_model(model_type, task, checkpoint_file, model_config_file, from_pretrained)
    datamodule = registry.get_task_datamodule(task, data_dir, batch_size, max_seq_len, num_workers, seqvec_type,
                                              model.config.tokenizer, **optional_dataset_args)
    model.config.optimizer = optimizer
    model.config.learning_rate = learning_rate
    model.config.warmup_steps = warmup_steps
    datamodule.setup()
    model.config.total_steps = int(len(datamodule.train_dataloader()) * num_train_epochs * train_frac)
    del datamodule
    datamodule = registry.get_task_datamodule(task, data_dir, batch_size, max_seq_len, num_workers, seqvec_type,
                                              model.config.tokenizer, **optional_dataset_args)
    datamodule.distributed = n_gpus > 1 or use_tpu
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
    lr_logger = lr_monitor.LearningRateMonitor(logging_interval='step')
    trainer_args['callbacks'] = [lr_logger]
    if not no_cuda and n_gpus > 1:
        trainer_args['gpus'] = n_gpus
        trainer_args['accelerator'] = 'ddp'
        trainer_args['replace_sampler_ddp'] = False
        # trainer_args['plugins'] = ['ddp_sharded']
    elif not no_cuda:
        trainer_args['gpus'] = 1
    trainer = pl.Trainer(**trainer_args)
    return datamodule, model, trainer


def run_eval(model_type: str,
             task: str,
             batch_size: int = 1024,
             num_log_iter: int = 20,
             fp16: bool = False,
             exp_name: typing.Optional[str] = None,
             from_pretrained: typing.Optional[str] = None,
             checkpoint_file: typing.Optional[str] = None,
             log_dir: str = './logs',
             model_config_file: typing.Optional[str] = None,
             data_dir: str = './data',
             output_dir: str = './results',
             use_tpu: bool = True,
             no_cuda: bool = False,
             n_gpus: int = 1,
             seed: int = 42,
             num_workers: int = 8,
             max_seq_len: int = 512,
             val_frac: float = 1.,
             seqvec_type: str = 'seqvec',
             metrics: typing.Union[list, tuple] = (),
             split: str = 'test',
             ) -> typing.Dict[str, float]:
    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir) / exp_dir

    # save all the hidden parameters.
    save_path.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(seed)
    model = registry.get_task_model(model_type, task, checkpoint_file, model_config_file, from_pretrained)

    datamodule = registry.get_task_datamodule(task, data_dir, batch_size, max_seq_len, num_workers, seqvec_type,
                                              model.config.tokenizer)

    datamodule.distributed = n_gpus > 1 or use_tpu

    evaluator_args = {
        'log_every_n_steps': num_log_iter,
        'default_root_dir': save_path,
        'limit_val_batches': val_frac,
    }
    if fp16:
        evaluator_args['precision'] = 16
    evaluator_args['logger'] = pl_loggers.TensorBoardLogger(log_dir, name=exp_name)

    if not no_cuda and n_gpus > 1:
        evaluator_args['gpus'] = n_gpus
        evaluator_args['accelerator'] = 'ddp'
        evaluator_args['replace_sampler_ddp'] = False
    elif not no_cuda:
        evaluator_args['gpus'] = 1

    trainer = pl.Trainer(**evaluator_args)

    results = trainer.test(model, test_dataloaders=datamodule.get_dataloader(split=split))

    with (save_path / 'results.pkl').open('wb') as f:
        pkl.dump(results, f)

    return results


'''
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
