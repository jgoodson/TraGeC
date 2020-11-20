from pathlib import Path
from typing import Dict, Type, Callable, Optional, Union

from .models.modeling import GeCModel, GeCDataModule

import json

PathType = Union[str, Path]


class GeCTaskSpec(object):
    """
    Attributes
    ----------
    name (str):
        The name of the GeC task
    datamodule (Type[GeCDataModule]):
        The datamodule used in the GeC task
    num_labels (int):
        number of labels used if this is a classification task
    models (Dict[str, GeCModel]):
        The set of models that can be used for this task. Default: {}.
    """

    def __init__(self,
                 name: str,
                 datamodule: Type[GeCDataModule],
                 num_labels: int = -1,
                 models: Optional[Dict[str, Type[GeCModel]]] = None):
        self.name = name
        self.datamodule = datamodule
        self.num_labels = num_labels
        self.models = models if models is not None else {}

    def register_model(self, model_name: str, model_cls: Optional[Type[GeCModel]] = None):
        if model_cls is not None:
            if model_name in self.models:
                raise KeyError(
                    f"A model with name '{model_name}' is already registered for this task")
            self.models[model_name] = model_cls
            return model_cls
        else:
            return lambda model_cls: self.register_model(model_name, model_cls)

    def get_model(self, model_name: str) -> Type[GeCModel]:
        return self.models[model_name]

class Registry:
    r"""Class for registry object which acts as the
    central repository for tragec."""

    task_name_mapping: Dict[str, GeCTaskSpec] = {}
    metric_name_mapping: Dict[str, Callable] = {}

    @classmethod
    def register_task(cls,
                      task_name: str,
                      num_labels: int = -1,
                      datamodule: Optional[Type[GeCDataModule]] = None,
                      models: Optional[Dict[str, Type[GeCModel]]] = None):
        """ Register a a new tragec task. This creates a new GeCTaskSpec.

        Args:

            task_name (str): The name of the tragec task.
            num_labels (int): Number of labels used if this is a classification task. If this
                is not a classification task, simply leave the default as -1.
            datamodule (Type[GeCDataModule]): The data module used in the tragec task.
            models (Optional[Dict[str, GeCModel]]): The set of models that can be used for
                this task. If you do not pass this argument, you can register models to the task
                later by using `registry.register_task_model`. Default: {}.

        """
        if datamodule is not None:
            if models is None:
                models = {}
            task_spec = GeCTaskSpec(task_name, datamodule, num_labels, models)
            return cls.register_task_spec(task_name, task_spec).datamodule
        else:
            return lambda datamodule: cls.register_task(task_name, num_labels, datamodule, models)

    @classmethod
    def register_task_spec(cls, task_name: str, task_spec: Optional[GeCTaskSpec] = None):
        """ Registers a task_spec directly. If you find it easier to actually create a
            GeCTaskSpec manually, and then register it, feel free to use this method,
            but otherwise it is likely easier to use `registry.register_task`.
        """
        if task_spec is not None:
            if task_name in cls.task_name_mapping:
                raise KeyError(f"A task with name '{task_name}' is already registered")
            cls.task_name_mapping[task_name] = task_spec
            return task_spec
        else:
            return lambda task_spec: cls.register_task_spec(task_name, task_spec)

    @classmethod
    def register_task_model(cls,
                            task_name: str,
                            model_name: str,
                            model_cls: Optional[Type[GeCModel]] = None):
        r"""Register a specific model to a task with the provided model name.
            The task must already be in the registry - you cannot register a
            model to an unregistered task.

        Args:
            task_name (str): Name of task to which to register the model.
            model_name (str): Name of model to use when registering task, this
                is the name that you will use to refer to the model on the
                command line.
            model_cls (Type[GeCModel]): The model to register.

        Examples:

        As with `registry.register_task`, this can both be used as a regular
        python function, and as a decorator. For example this:

            class ProteinBertForSequenceToSequenceClassification():
                ...
            registry.register_task_model(
                'secondary_structure', 'transformer',
                ProteinBertForSequenceToSequenceClassification)

        and as a decorator:

            @registry.register_task_model('secondary_structure', 'transformer')
            class ProteinBertForSequenceToSequenceClassification():
                ...

        are both equivalent.
        """
        if task_name not in cls.task_name_mapping:
            raise KeyError(
                f"Tried to register a task model for an unregistered task: {task_name}. "
                f"Make sure to register the task {task_name} first.")
        return cls.task_name_mapping[task_name].register_model(model_name, model_cls)

    @classmethod
    def get_task_spec(cls, name: str) -> GeCTaskSpec:
        return cls.task_name_mapping[name]

    @classmethod
    def get_metric(cls, name: str) -> Callable:
        return cls.metric_name_mapping[name]

    @classmethod
    def get_task_model(cls,
                       model_name: str,
                       task_name: str,
                       config_file: Optional[PathType] = None,
                       load_dir: Optional[PathType] = None) -> GeCModel:
        """ Create a tragec task model, either from scratch or from a pretrained model.
            This is mostly a helper function that evaluates the if statements in a
            sensible order if you pass all three of the arguments.
        Args:
            model_name (str): Which type of model to create (e.g. transformer, lstm, ...)
            task_name (str): The tragec task for which to create a model
            config_file (str, optional): A json config file that specifies hyperparameters
            load_dir (str, optional): A save directory for a pretrained model
        Returns:
            model (GeCModel): A tragec task model
        """
        task_spec = registry.get_task_spec(task_name)
        model_cls = task_spec.get_model(model_name)

        if load_dir is not None:
            model = model_cls.from_pretrained(load_dir, num_labels=task_spec.num_labels)
        else:
            config_class = model_cls.config_class
            if config_file is not None:
                # TODO: consider re-writing the TAPE config base class to work with multiple inheritance
                with open(config_file, "r", encoding='utf-8') as reader:
                    text = reader.read()
                config = config_class(**json.loads(text))
            else:
                config = config_class()
            config.num_labels = task_spec.num_labels
            model = model_cls(config)
        return model

    @classmethod
    def get_task_datamodule(cls,
                            task_name: str,
                            data_dir: str,
                            batch_size: int,
                            seqvec_type: str,
                            max_seq_len: int,
                            num_workers: int,
                            **kwargs) -> GeCDataModule:
        """ Create a tragec task model, either from scratch or from a pretrained model.
            This is mostly a helper function that evaluates the if statements in a
            sensible order if you pass all three of the arguments.
        Args:
            task_name (str): The tragec task for which to create a model
        Returns:
            module (GeCDataModule): A tragec task data module
        """
        task_spec = registry.get_task_spec(task_name)
        return task_spec.datamodule(data_dir, batch_size, seqvec_type, max_seq_len, num_workers,
                                    **kwargs)


registry = Registry()
