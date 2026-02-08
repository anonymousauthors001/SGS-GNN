from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import torch


def select_two_classes(labels, class_pair: Optional[Iterable[int]] = None) -> Tuple[int, int]:
    if class_pair is not None:
        pair = tuple(class_pair)
        if len(pair) != 2:
            raise ValueError("class_pair must contain exactly two class labels.")
        return int(pair[0]), int(pair[1])

    if torch.is_tensor(labels):
        unique = torch.unique(labels.detach().cpu()).tolist()
    else:
        unique = np.unique(np.asarray(labels)).tolist()

    unique = sorted(int(u) for u in unique)
    if len(unique) < 2:
        raise ValueError("Need at least two classes to visualize.")

    return unique[0], unique[1]
