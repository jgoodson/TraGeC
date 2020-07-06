from typing import Sequence, Union

import numpy as np
import scipy.stats

from .registry import registry


@registry.register_metric('mse')
def mean_squared_error(target: Sequence[Union[float, np.ndarray]],
                       prediction: Sequence[Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


@registry.register_metric('mae')
def mean_absolute_error(target: Sequence[Union[float, np.ndarray]],
                        prediction: Sequence[Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(target: Sequence[Union[float, np.ndarray]],
              prediction: Sequence[Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    # noinspection PyTypeChecker
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[Union[float, np.ndarray]], Sequence[Sequence[Union[float, np.ndarray]]]]) -> \
        Union[float, np.ndarray]:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array: np.ndarray = np.asarray(label)
            pred_array: np.ndarray = np.asarray(score).argmax(-1)
            mask: np.ndarray = label_array != -1
            # noinspection PyTypeChecker
            is_correct: np.ndarray = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total
