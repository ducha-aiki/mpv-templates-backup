from typing import Callable

import numpy as np
from dataclasses import dataclass, field
DBL_EPS = np.finfo(float).eps


@dataclass
class RansacPlanarLog:
    """ future tip: make it comparable as used in update """
    H: np.array = field(default=None, repr=False)
    mask: np.array = field(default=None, repr=False)
    n: int = 0
    loss: float = np.inf
    i: int = 0

    def __post_init__(self):
        if self.mask is not None:
            self.n = self.mask.sum().item()

    def __lt__(self, other):
        if self.n != other.n:
            return self.n < other.n
        return self.loss-DBL_EPS > other.loss

    def __gt__(self, other):
        return other < self

    def __eq__(self, other):
        return not (self < other or other < self)


@dataclass
class RansacPlanarParams:
    sample_size: int = 4
    ransac_th: float = 15.0  # pixel threshold for correspondence to be counted as inlier
    max_iter: int = 10000    # maximum iteration, overrides confidence
    conf: float = 0.99       # confidence


@dataclass(eq=False)
class RansacPlanarFunctions:
    nsamples: Callable = None
    getH: Callable = None                   # estimates homography from minimal sample
    hdist: Callable = None                  # calculates one - way reprojection error
    sampler: Callable = None                # draws random sample from pts_matches
    check_sample: Callable = None           # is a set of samples valid for homography estimation?
