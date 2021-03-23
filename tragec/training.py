import logging
import os
import typing
from pathlib import Path
import json
from time import strftime, gmtime
import random

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import lr_monitor

from .registry import registry

logger = logging.getLogger(__name__)


def generate_expname_and_save_dir(exp_name: typing.Optional[str], task: str, model_type: str, output_dir: str):
    if exp_name is None:
        time_stamp = strftime("%y-%m-%d-%H-%M-%S", gmtime())
        exp_name = f"{task}_{model_type}_{time_stamp}_{random.randint(0, int(1e6)):0>6d}"
    return exp_name, Path(output_dir) / exp_name


def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              fp16_backend: str = 'native',
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
              fast_dev_run: bool = False,
              sharded_training: bool = False,
              deepspeed: bool = False,
              ) -> None:
    input_args = locals()

    exp_name, save_path = generate_expname_and_save_dir(exp_name, task, model_type, output_dir)

    save_path.mkdir(parents=True, exist_ok=True)
    with (save_path / 'args.json').open('w') as f:
        json.dump(input_args, f)

    optional_dataset_args = {
        'percentmasked': percentmasked,
    }

    if resume_from_checkpoint:
        if not checkpoint_file:
            last_version = sorted([f.split('_')[1] for f in os.listdir(f'{log_dir}/{exp_name}')
                                   if f.startswith('version_')], key=int)[-1]

            checkpoints = [f for f in os.listdir(f'{log_dir}/{exp_name}/version_{last_version}/checkpoints')
                           if f.endswith('.ckpt')]
            if checkpoints:
                checkpoint_file = f'{log_dir}/{exp_name}/version_{last_version}/checkpoints/{checkpoints[0]}'
            else:
                raise UserWarning('Did not find checkpoint from most recent run. Starting training from scratch.')

    datamodule, model, trainer = prepare_trainer(batch_size, checkpoint_file, data_dir, eval_freq, exp_name, fp16,
                                                 fp16_backend, from_pretrained, gradient_accumulation_steps,
                                                 learning_rate, log_dir, max_grad_norm, max_seq_len, model_config_file,
                                                 model_type, n_gpus, no_cuda, num_log_iter, num_train_epochs,
                                                 num_workers, optimizer, patience, optional_dataset_args, save_path,
                                                 seed, seqvec_type, task, train_frac, use_tpu, val_frac, warmup_steps,
                                                 fast_dev_run, sharded_training, deepspeed)

    trainer.fit(model=model, datamodule=datamodule)

    model.save_pretrained(save_path)


def prepare_trainer(batch_size, checkpoint_file, data_dir, eval_freq, exp_name, fp16, fp16_backend, from_pretrained,
                    gradient_accumulation_steps, learning_rate, log_dir, max_grad_norm, max_seq_len, model_config_file,
                    model_type, n_gpus, no_cuda, num_log_iter, num_train_epochs, num_workers, optimizer, patience,
                    optional_dataset_args, save_path, seed, seqvec_type, task, train_frac, use_tpu, val_frac,
                    warmup_steps, fast_dev_run, sharded_training, deepspeed):
    pl.seed_everything(seed)
    model = registry.get_task_model(model_type, task, checkpoint_file, model_config_file, from_pretrained)
    datamodule = registry.get_task_datamodule(task, data_dir, batch_size // gradient_accumulation_steps, max_seq_len,
                                              num_workers, seqvec_type,
                                              model.config.tokenizer, **optional_dataset_args)
    model.config.optimizer = optimizer
    model.config.learning_rate = learning_rate
    model.config.warmup_steps = warmup_steps // gradient_accumulation_steps
    datamodule.setup()
    model.config.total_steps = int(len(datamodule.train_dataloader()) *
                                   num_train_epochs *
                                   train_frac /
                                   gradient_accumulation_steps)
    del datamodule
    datamodule = registry.get_task_datamodule(task, data_dir, batch_size // gradient_accumulation_steps, max_seq_len,
                                              num_workers, seqvec_type=seqvec_type, tokenizer=model.config.tokenizer,
                                              **optional_dataset_args)
    datamodule.distributed = n_gpus > 1 or use_tpu
    trainer_kwargs = {
        'accumulate_grad_batches': gradient_accumulation_steps,
        'max_epochs': num_train_epochs,
        'log_every_n_steps': num_log_iter,
        'default_root_dir': save_path,
        'limit_train_batches': train_frac,
        'limit_val_batches': val_frac,
        'val_check_interval': eval_freq,
        'gradient_clip_val': max_grad_norm,
        'fast_dev_run': fast_dev_run,
    }
    if patience != -1:
        es_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        trainer_kwargs['callbacks'] = [es_callback]
    if fp16:
        trainer_kwargs['precision'] = 16
        trainer_kwargs['amp_backend'] = fp16_backend
    if checkpoint_file:
        trainer_kwargs['resume_from_checkpoint'] = checkpoint_file
    trainer_kwargs['logger'] = pl_loggers.TensorBoardLogger(log_dir, name=exp_name)
    lr_logger = lr_monitor.LearningRateMonitor(logging_interval='step')
    trainer_kwargs['callbacks'] = [lr_logger]
    if not no_cuda and n_gpus > 1:
        trainer_kwargs['gpus'] = n_gpus
        trainer_kwargs['accelerator'] = 'ddp'
        trainer_kwargs['replace_sampler_ddp'] = False
        trainer_kwargs['plugins'] = []
        if sharded_training:
            trainer_kwargs['plugins'].append('ddp_sharded')
        if deepspeed:
            trainer_kwargs['plugins'].append('deepspeed')
    elif not no_cuda:
        trainer_kwargs['gpus'] = 1
    trainer = pl.Trainer(**trainer_kwargs)
    return datamodule, model, trainer


def run_eval(model_type: str,
             task: str,
             batch_size: int = 1024,
             fp16: bool = False,
             fp16_backend: str = 'native',
             exp_name: typing.Optional[str] = None,
             from_pretrained: typing.Optional[str] = None,
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
             split: typing.Optional[str] = None,
             ) -> typing.Dict[str, float]:
    exp_name, save_path = generate_expname_and_save_dir(exp_name, task, model_type, output_dir)

    pl.seed_everything(seed)
    model = registry.get_task_model(model_type, task, None, model_config_file, from_pretrained)

    datamodule = registry.get_task_datamodule(task, data_dir, batch_size, max_seq_len, num_workers, seqvec_type,
                                              model.config.tokenizer)

    datamodule.distributed = n_gpus > 1 or use_tpu

    evaluator_kwargs = {
        'default_root_dir': save_path,
        'limit_test_batches': val_frac,
    }
    if fp16:
        evaluator_kwargs['precision'] = 16
        evaluator_kwargs['amp_backend'] = fp16_backend
    evaluator_kwargs['logger'] = pl_loggers.TensorBoardLogger(log_dir, name=exp_name)

    if not no_cuda and n_gpus > 1:
        evaluator_kwargs['gpus'] = n_gpus
        evaluator_kwargs['accelerator'] = 'ddp'
        evaluator_kwargs['replace_sampler_ddp'] = False
    elif not no_cuda:
        evaluator_kwargs['gpus'] = 1

    trainer = pl.Trainer(**evaluator_kwargs)

    if split:
        datamodule.setup()
        results = trainer.test(model, test_dataloaders=datamodule.get_dataloader(split=split))
    else:
        results = trainer.test(model, datamodule=datamodule)

    with (save_path / 'results.json').open('wb') as f:
        json.dump(results, f)

    return results
