import json
from pathlib import Path
from typing import Dict, Type, Callable, Optional, Union

from .models.modeling import BioModel
from .tasks.tasks import BioDataModule

PathType = Union[str, Path]


class TaskSpec(object):
    """
    Attributes
    ----------
    name (str):
        The name of the GeC task
    datamodule (Type[BioDataModule]):
        The datamodule used in the GeC task
    num_labels (int):
        number of labels used if this is a classification task
    models (Dict[str, BioModel]):
        The set of models that can be used for this task. Default: {}.
    """

    def __init__(self, name: str, datamodule: Type[BioDataModule], num_labels: int = -1,
                 models: Optional[Dict[str, Type[BioModel]]] = None, model_kwargs: Optional[dict] = None):
        self.name = name
        self.datamodule = datamodule
        self.num_labels = num_labels
        self.models = models if models is not None else {}
        self.extra_conf_kwargs = model_kwargs

    def register_model(self, model_name: str, model_cls: Optional[Type[BioModel]] = None):
        if model_cls is not None:
            if model_name in self.models:
                raise KeyError(
                    f"A model with name '{model_name}' is already registered for this task")
            self.models[model_name] = model_cls
            return model_cls
        else:
            return lambda model_cls: self.register_model(model_name, model_cls)

    def get_model(self, model_name: str) -> Type[BioModel]:
        try:
            return self.models[model_name]
        except KeyError:
            raise KeyError(f'{model_name} not found in registry for task {self.name}. Options: {self.models.keys()}')


class Registry:
    r"""Class for registry object which acts as the
    central repository for tragec."""

    task_name_mapping: Dict[str, TaskSpec] = {}
    metric_name_mapping: Dict[str, Callable] = {}

    @classmethod
    def register_task(cls,
                      task_name: str,
                      num_labels: int = -1,
                      datamodule: Optional[Type[BioDataModule]] = None,
                      models: Optional[Dict[str, Type[BioModel]]] = None,
                      **model_kwargs):
        """ Register a a new tragec task. This creates a new TaskSpec.

        Args:

            task_name (str): The name of the tragec task.
            num_labels (int): Number of labels used if this is a classification task. If this
                is not a classification task, simply leave the default as -1.
            datamodule (Type[BioDataModule]): The data module used in the tragec task.
            models (Optional[Dict[str, BioModel]]): The set of models that can be used for
                this task. If you do not pass this argument, you can register models to the task
                later by using `registry.register_task_model`. Default: {}.

        """
        if datamodule is not None:
            if models is None:
                models = {}
            task_spec = TaskSpec(task_name, datamodule, num_labels, models, model_kwargs)
            return cls.register_task_spec(task_name, task_spec).datamodule
        else:
            return lambda datamodule: cls.register_task(task_name, num_labels, datamodule, models, **model_kwargs)

    @classmethod
    def register_task_spec(cls, task_name: str, task_spec: Optional[TaskSpec] = None):
        """ Registers a task_spec directly. If you find it easier to actually create a
            TaskSpec manually, and then register it, feel free to use this method,
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
                            model_cls: Optional[Type[BioModel]] = None):
        r"""Register a specific model to a task with the provided model name.
            The task must already be in the registry - you cannot register a
            model to an unregistered task.

        Args:
            task_name (str): Name of task to which to register the model.
            model_name (str): Name of model to use when registering task, this
                is the name that you will use to refer to the model on the
                command line.
            model_cls (Type[BioModel]): The model to register.

        Examples:

        As with `registry.register_task`, this can both be used as a regular
        python function, and as a decorator. For example this:

            class GeCBertForSequenceToSequenceClassification():
                ...
            registry.register_task_model(
                'secondary_structure', 'transformer',
                GeCBertForSequenceToSequenceClassification)

        and as a decorator:

            @registry.register_task_model('secondary_structure', 'transformer')
            class GeCBertForSequenceToSequenceClassification():
                ...

        are both equivalent.
        """
        if task_name not in cls.task_name_mapping:
            raise KeyError(
                f"Tried to register a task model for an unregistered task: {task_name}. "
                f"Make sure to register the task {task_name} first.")
        return cls.task_name_mapping[task_name].register_model(model_name, model_cls)

    @classmethod
    def get_task_spec(cls, name: str) -> TaskSpec:
        return cls.task_name_mapping[name]

    @classmethod
    def get_metric(cls, name: str) -> Callable:
        return cls.metric_name_mapping[name]

    @classmethod
    def get_task_model(cls,
                       model_name: str,
                       task_name: str,
                       checkpoint: Optional[PathType] = None,
                       config_file: Optional[PathType] = None,
                       pretrained_model: Optional[PathType] = None) -> BioModel:
        """ Create a tragec task model, either from scratch or from a pretrained model.
            This is mostly a helper function that evaluates the if statements in a
            sensible order if you pass all three of the arguments.
        Args:
            model_name (str): Which type of model to create (e.g. transformer, lstm, ...)
            task_name (str): The tragec task for which to create a model
            checkpoint (str, optional): A pytorch-lightning checkpoint of the model
            config_file (str, optional): A json config file that specifies hyperparameters
            pretrained_model (str, optional): A save directory for a pretrained model or save name
        Returns:
            model (BioModel): A tragec task model
        """
        task_spec = registry.get_task_spec(task_name)
        model_cls = task_spec.get_model(model_name)

        if pretrained_model is not None:
            model = model_cls.from_pretrained(pretrained_model, num_labels=task_spec.num_labels,
                                              **task_spec.extra_conf_kwargs)
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
            for k, v in task_spec.extra_conf_kwargs.items():
                setattr(config, k, v)
            if checkpoint is not None:
                model = model_cls.load_from_checkpoint(checkpoint, config=config)
            else:
                model = model_cls(config)
        return model

    @classmethod
    def get_task_datamodule(cls,
                            task_name: str,
                            data_dir: str,
                            batch_size: int,
                            max_seq_len: int,
                            num_workers: int,
                            seqvec_type: str,
                            tokenizer: str,
                            **kwargs) -> BioDataModule:
        task_spec = registry.get_task_spec(task_name)
        return task_spec.datamodule(data_dir, batch_size, max_seq_len, num_workers, seqvec_type, tokenizer,
                                    **kwargs)


registry = Registry()
