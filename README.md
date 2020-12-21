# TraGec
## **Tra**nsformers for **G**ene **C**luster Learning

## Overview
This is a Python package heavily inspired by and based on the code of [TAPE](https://github.com/songlab-cal/tape), Tasks Assessing Protein Embeddings.

TraGeC extends tape to include models and tasks not just for protein sequence learning but also for genome-level gene cluster sequences. This is intended to studying deep learning approaches to identifying, classifying, and otherwise studying bacterial gene clusters. 

## Background

Much like protein sequences, bacterial (and often fungal) genes are arranged in a two-dimensional sequence, and due to quirks of bacterial evolution, are frequently found organized in units such as operons or gene clusters. While in general the behavior of a gene product is primarily due to its sequence, determining the true function of a particular gene from only its sequence is an extremely challenging task. Think of DNA-binding regulatory proteins. Very subtle changes in the sequence of a particular HTH DNA-binding protein can dramatically change its regulatory targets. Frequently, bacterial genes are organized into functional groups, with gene clusters containing the majority of genes necessary for a particular function found together. Often, a regulatory gene responsible for activation of a gene cluster is found adjacent to those genes it regulates. In this way, we can often extract information about genes from the genomic context in which they are found. 

One example of using natural language modeling-inspired tools for gene cluster analysis is [DeepBGC](https://github.com/Merck/deepbgc), a tool using bi-direction LSTM models to model bacterial or fungal genome organization. DeepBGC is intended to identify biosynthetic gene clusters based on the domain content of each gene and the organization of those genes in the genome. Protein domains are simple to identify using hidden Markov models (HMMs) and much of the information about the function of a particular gene can be obtained from identifying its component domains. 

TraGeC is intended to be a synthesis of these two levels of deep learning modeling of biological sequences. While protein domains are important, a great deal of information can be lost by condensing all of the rich information found in a protein sequence to a simple list of domains. Importantly, many proteins important to some types of gene clusters have domains which are not present in the most commonly used domain database, [PFAM](http://pfam.xfam.org/). TAPE (as well as the many excellent papers and methods written about protein sequence deep learning, many of which are collected [here](https://github.com/yangkky/Machine-learning-for-proteins)) contains methods to condense all of the sequence information to a convenient "embedded" representation which can be used for different biologically-motivated learning tasks. We propose to use these protein embeddings as input to a genome-level sequence analysis using deep learning methods pioneered for natural language modeling, essentially replacing the normal embedding step used for token-level input with a protein-level deep learning embedding. 

Both protein-level and genome-level biological sequence learning tasks are surprisingly similar, relying on biological sequences for input where individual units contain a great deal of context-specific meaning. In the case of genetics, these two levels are nested, with the protein sequences (made of individual amino acids) forming (one type of) the individual units of a genome (made up of individual proteins). For this reason, we have included both protein-level and genome-level learning methods in TraGeC to facilitate development and optimization of both the protein-level and genome-level learning models.

## Installation

This package is not available on PyPI or Anaconda. You can either download the reposity and run
```
python setup.py install
```
from the main `tragec` folder.

## Usage

TraGeC is structured very similarly to TAPE, itself using a similar model to the [HuggingFace Transformers library](https://huggingface.co/transformers/index.html). TraGeC contains three command-line tools, `tragec-train`, `tragec-eval`, and `tragec-infer`. Unlike the current publicly-available version of TAPE, TraGeC has been re-written to utilize PyTorch-Lightning, greatly simplifying its development and usage and eliminating dependencies on Nvidia Apex and seperate distributed training methods. 

All commands require a model type and a task name. Currently, the available tasks include: `embed`, `masked_language_modeling` (pre-training for protein-level tasks), `masked_recon_modeling` (pre-training for cluster-level tasks), `remote_homology`, `fluorescence`, `stability`, `contact_prediction`, `secondary_structure` (all protein-level), and `classify_gec` (cluster-level). Training, evaluation, and inference are split into three distinct commands.

### tragec-train

In general, models may be either pre-trained or fine-tuned (depending on the particular task) using `tragec-train`. All tasks are available for this function except `embed`. Tasks are pre-defined and define the available data, data splits, model behavior, and training and evaluation logic. More detail on other command-line flags to come.

### tragec-eval

Once a model has been trained, `tragec-eval` is available to run the model in evaluation mode on a testing dataset and evaluate its behavior. The metrics and output are specified in the task code itself and may not be changed at run-time. To perform custom testing, you should use `tragec-infer` to generate model output and process it externally.

### tragec-infer

The final command, `tragec-infer`, allows for output of the task models themselves to obtain embeddings, predictions, classifications, or whatever type of output the specified task model provides.

## Development

TraGeC is designed to facilitate (relatively) easy integration of new models or tasks and to allow both models and tasks to be shared between protein- and cluster/genome-level tasks. Base models are defined in individual source files in `tragec/models/` while general task types (separated by modeling head type) are defined in source files in `tragec/tasks`. Raw data access is largely defined in the `tragec/datasets.py` file and organized in the individual task specifications. 

### Models

Models consist of base models defining a PyTorch (-Lightning) module which accept sequence representations and output sequence embeddings. All models follow normal PyTorch conventions and accept batched input. For protein-level models, the sequence representation is tokenized amino-acid sequences, shape: `[batch_size, sequence_length]`. For cluster/genome-level models the sequence representation is pre-computed protein embeddings, shape: `[batch_size, sequence_length, rep_dimension]`. The input is always padded to equal length (required for tensors) in the `DataSet` itself before it is provided to the model. These models should provide (at least) two outputs, sequence output and pooled output. Sequence output should be of shape `[batch_size, sequence_length, hidden_size]` while pooled output should be `[batch_size, hidden_size]` and may be generated in any way. The `forward` method should return a tuple containing these two outputs as the first two items.

Base models should be defined according to the prototype in `tragec/models/models_bert.py` and be based off of the class `BioModel` in `tragec/modeling`. This PyTorch-Lightning `LightningModule` subclass defines everything necessary for integration with all of the TraGeC task models. In general, you should define an abstract class, subclassing BioModel  and then two specific classes of the base model for both protein-level and genome-level input. **TODO: more information about this.** 

Two embedding modules are provided `ProteinEmbedding` and `GeCEmbeddings` which take as input the input data from the two types of datasets and provide `[batch_size, sequence_length, hidden_size]` shape output tensors which are suitable for input to most of the HuggingFace Transformers models using the `inputs_embeds` argument. These may be (but don't need to be) used to adapt many generic NLM models for use as a base model in TraGeC. 

### Tasks

If a protein- or genome-level base model has been created according to the above specifications (subclasses BioModel, defines a forward function that accepts the appropriate `sequence_rep` input, outputs a tuple of `sequence_output`, `pooled_output`), it can be automatically adapted to all of the available tasks using the tragec.tasks.registry.create_and_register_models function. This function is designed to automatically add all available output heads for the two base models provided with the base abstract class. 

Tasks themselves are defined in parts. The final model is created using multiple inheritence combining a base model and a task model. First, a subclass of BioModel is created which assumes that `self.model` contains a suitable base model. This task model should define a `forward` function with a signature `forward(sequence_rep, input_mask, **kwargs) -> Tuple` that defines whatever method is necessary to go from sequence or pooled embedding output to task-specific output/predictions/etc. The first argument is always provided, the second is currently provided by all datasets but is not guarantees, and optional keyword arguments must be accepted but not used unless defined in the task-specific DataSet input. The model should also define a `_compare` function which accepts the output of the `forward` function and a `batch` argument with the original DataSet output (input to the model, which should contain the targets for \[semi-\]supervised training) that returns a 2-tuple of `(loss, metrics)` where `loss` is a loss tensor to be used for model training and `metrics` is a dictionary containing metrics for logging and evaluation. You may create one of these per general class of task (different single-value prediction tasks share the same code for example)

The second component of a task is the DataModule. This should subclass BioDataModule and define five attributes, the underlying DataSet class `dataset`, the name of the different data splits `split_names` which can be provided to the dataset to obtain a particular subset of data, and finally the split names that represent the training, validation, and testing splits (`train_split`, `val_split`, and `test_split`). This DataModule should be registered with the TraGeC task/model registry using the `@registry.register_task` decorator. You should create one DatModule for each specific task which needs a specific data source.

The final element for a task is to provide a function that accepts a base class, base model class, name, and sequence type and creates task-specific models for each specific task. Just look at the example in `tragec/tasks/task_singleclass.py` as an example. This should discriminate between `'prot'` tasks and `'gec'` tasks and name the model accordingly (`prot_bert` vs `gec_bert`, for example). This function should then be added to the `tragec/tasks/registry.py` module for automatic task registration.

### Datasets

The `DataModule`s define which DataSets are used for a particular task, but the actual access and processing of the data is defined in `tragec/datasets.py`. Datasets should be normal PyTorch Datasets subclassed from the `BioDataset` abstract base class.
 
These must define:
- A constructor that accepts a path as the first argument, the name of the data split as the second argument, with additional arguments optional as necessary which will be passed through from the DataModule.
- A `__len__` function
- A `__getitem__` function that accepts integers from 0 to len(Dataset) and returns individual sequences 
- A `collate_fn` function that accepts individual batches as a list of single objects (whatever is returned from `__getitem__`). This function should do whatever processing is necessary to convert the input to PyTorch tensors, batch together multiple items into single-batch tensors for the model, and return a dictionary with, at minimum `sequence_rep` and `targets` keys, optionally with `input_mask` and whichever other names correspond to arguments that your task model uses. This function is where any input should be massaged into the exact format the task-specific model is expecting.

These datasets should be pickleable. All of the currently-defined tasks define datasets that use LMDB databases with the LMDBDataset class that defines `__getstate__` and `__setstate__` methods to allow these database environment objects to be pickled by re-opening them when unpickled. 