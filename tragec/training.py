import logging
import os
import typing
from pathlib import Path
import json
from time import strftime, gmtime
import random
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import lr_monitor
from pytorch_lightning.plugins import DDPPlugin

from .registry import registry

logger = logging.getLogger(__name__)


def generate_expname_and_save_dir(exp_name: typing.Optional[str], task: str, model_type: str, output_dir: str):
    if exp_name is None:
        time_stamp = strftime("%y-%m-%d-%H-%M-%S", gmtime())
        exp_name = f"{task}_{model_type}_{time_stamp}_{random.randint(0, int(1e6)):0>6d}"
    return exp_name, Path(output_dir) / exp_name


def process_dataset_kwargs(args: argparse.Namespace) -> typing.Dict:
    eff_batch_size = args.batch_size // args.gradient_accumulation_steps

    dataset_args = {
        'task_name': args.task,
        'data_dir': args.data_dir,
        'batch_size': eff_batch_size,
        'max_seq_len': args.max_seq_len,
        'num_workers': args.num_workers,
        'percentmasked': args.percentmasked,
        'seqvec_type': args.seqvec_type,
        'tokenizer': args.tokenizer,
    }
    return dataset_args


def process_trainer_kwargs(args: argparse.Namespace,
                           exp_name: str,
                           checkpoint_file: str) -> typing.Dict:
    trainer_kwargs = {
        'accumulate_grad_batches': args.gradient_accumulation_steps,
        'max_epochs': args.num_train_epochs,
        'log_every_n_steps': args.num_log_iter,
        'default_root_dir': args.save_path,
        'limit_train_batches': args.train_frac,
        'limit_val_batches': args.val_frac,
        'val_check_interval': args.eval_freq,
        'gradient_clip_val': args.max_grad_norm,
        'fast_dev_run': args.fast_dev_run,
    }
    if args.patience != -1:
        es_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)
        trainer_kwargs['callbacks'] = [es_callback]
    if args.fp16:
        trainer_kwargs['precision'] = 16
        trainer_kwargs['amp_backend'] = args.fp16_backend
    if checkpoint_file:
        trainer_kwargs['resume_from_checkpoint'] = checkpoint_file
    trainer_kwargs['logger'] = pl_loggers.TensorBoardLogger(args.log_dir, name=exp_name)
    lr_logger = lr_monitor.LearningRateMonitor(logging_interval='step')
    trainer_kwargs['callbacks'] = [lr_logger]
    if not args.no_cuda and args.n_gpus > 1:
        trainer_kwargs['gpus'] = args.n_gpus
        trainer_kwargs['accelerator'] = 'ddp'
        trainer_kwargs['replace_sampler_ddp'] = False
        trainer_kwargs['plugins'] = [DDPPlugin(find_unused_parameters=False), ]
        if args.sharded_training:
            trainer_kwargs['plugins'].append('ddp_sharded')
        if args.deepspeed:
            trainer_kwargs['plugins'].append('deepspeed')
    elif not args.no_cuda:
        trainer_kwargs['gpus'] = 1
    return trainer_kwargs


def run_train(args: argparse.Namespace) -> None:
    """
    Runs training using PyTorch-Lightning.

    :param args: A Namespace with matching values to those prepared
    by the argparse module in main.py.
    """

    exp_name, save_path = generate_expname_and_save_dir(args.exp_name, args.task, args.model_type, args.output_dir)

    log_dir = args.log_dir

    save_path.mkdir(parents=True, exist_ok=True)
    with (save_path / 'args.json').open('w') as f:
        json.dump(args, f)

    checkpoint_file = args.checkpoint_file
    if args.resume_from_checkpoint:
        if not args.checkpoint_file:
            last_version = sorted([f.split('_')[1] for f in os.listdir(f'{log_dir}/{exp_name}')
                                   if f.startswith('version_')], key=int)[-1]

            checkpoints = [f for f in os.listdir(f'{log_dir}/{exp_name}/version_{last_version}/checkpoints')
                           if f.endswith('.ckpt')]
            if checkpoints:
                checkpoint_file = f'{log_dir}/{exp_name}/version_{last_version}/checkpoints/{checkpoints[0]}'
            else:
                raise UserWarning('Did not find checkpoint from most recent run. Starting training from scratch.')

    datamodule, model, trainer = prepare_trainer(args, exp_name, checkpoint_file)

    trainer.fit(model=model, datamodule=datamodule)

    model.save_pretrained(save_path)


def prepare_trainer(args: argparse.Namespace, exp_name: str, checkpoint_file: str):
    pl.seed_everything(args.seed)

    model = registry.get_task_model(model_name=args.model_type,
                                    task_name=args.task,
                                    checkpoint=checkpoint_file,
                                    config_file=args.model_config_file,
                                    pretrained_model=args.from_pretrained)

    dataset_kwargs = process_dataset_kwargs(args)

    datamodule = registry.get_task_datamodule(**dataset_kwargs)

    model.config.optimizer = args.optimizer
    model.config.learning_rate = args.learning_rate
    model.config.warmup_steps = args.warmup_steps // args.gradient_accumulation_steps

    # The only way to know how many total steps there will be is to interrogate the dataset.
    # We need to know that to configure our learning rate scheduler, so we have to actually
    # open up the underlying dataset and prepare it to find its length.
    datamodule.setup()
    model.config.total_steps = int(len(datamodule.train_dataloader()) *
                                   args.num_train_epochs *
                                   args.train_frac /
                                   args.gradient_accumulation_steps)
    # Doing this will break setup of the DataModule during training initialization so we
    # just start over fresh.
    del datamodule
    datamodule = registry.get_task_datamodule(**dataset_kwargs)

    datamodule.distributed = args.n_gpus > 1 or args.use_tpu

    trainer_kwargs = process_trainer_kwargs(args, exp_name, checkpoint_file)

    trainer = pl.Trainer(**trainer_kwargs)
    return datamodule, model, trainer


def process_eval_kwargs(args: argparse.Namespace, exp_name) -> typing.Dict:
    evaluator_kwargs = {
        'default_root_dir': args.save_path,
        'limit_test_batches': args.val_frac,
    }
    if args.fp16:
        evaluator_kwargs['precision'] = 16
        evaluator_kwargs['amp_backend'] = args.fp16_backend
    evaluator_kwargs['logger'] = pl_loggers.TensorBoardLogger(args.log_dir, name=exp_name)

    if not args.no_cuda and args.n_gpus > 1:
        evaluator_kwargs['gpus'] = args.n_gpus
        evaluator_kwargs['accelerator'] = 'ddp'
        evaluator_kwargs['replace_sampler_ddp'] = False
    elif not args.no_cuda:
        evaluator_kwargs['gpus'] = 1

    return evaluator_kwargs


def run_eval(args: argparse.Namespace) -> typing.Dict[str, float]:
    exp_name, save_path = generate_expname_and_save_dir(args.exp_name, args.task, args.model_type, args.output_dir)

    pl.seed_everything(args.seed)
    model = registry.get_task_model(model_name=args.model_type,
                                    task_name=args.task,
                                    config_file=args.model_config_file,
                                    pretrained_model=args.from_pretrained)

    dataset_kwargs = process_dataset_kwargs(args)

    datamodule = registry.get_task_datamodule(**dataset_kwargs)

    datamodule.distributed = args.n_gpus > 1 or args.use_tpu

    evaluator_kwargs = process_eval_kwargs(args, exp_name)

    trainer = pl.Trainer(**evaluator_kwargs)

    if args.split:
        datamodule.setup()
        results = trainer.test(model, test_dataloaders=datamodule.get_dataloader(split=args.split))
    else:
        results = trainer.test(model, datamodule=datamodule)

    with (save_path / 'results.json').open('wb') as f:
        json.dump(results, f)

    return results
