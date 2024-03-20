"""Systems of state matrix representation from the Appendix."""

import logging

from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def convm(x: ArrayLike, p: int) -> np.ndarray:
    """Construct the convolution matrix of the signal x with p number of parameters.

    (N + p - 1) by p non-symmetric Toeplitz matrix
    """
    _x = np.array(x, dtype=complex).ravel()
    if p < 1:
        raise ValueError(f"{p=} must be greater or equal to 1.")

    N = len(_x) + 2 * p - 2
    # the signal centered over its support
    # needed for the signal information-preserving frequency spectrum
    xcol = (_x.copy()).reshape(-1, 1)
    xpad = np.concatenate((np.zeros((p - 1, 1)), xcol, np.zeros((p - 1, 1))))
    X = np.empty([len(_x) + p - 1, p], dtype=complex)
    for i in range(p):
        X[:, i] = xpad[p - i - 1:N - i, 0]
    return X


def covar(x: ArrayLike, p: int) -> np.ndarray:
    """Covariance Matrix.

    p x p hermitian toeplitz matrix of sample covariances
    """
    _x = np.array(x, dtype=complex)
    m = len(_x)
    # remove the mean
    _x = _x - np.mean(_x)
    conv = convm(x, p + 1)
    R = conv.conjugate().transpose() @ conv / (m - 1)
    logging.warning(f'{conv.shape=}')
    logging.warning(f'{conv=}')

    return R


def nprony(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Normalized Prony Systems of Equations."""
    raise NotImplementedError()


def ywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Yuler-Walker Systems of Equations.

    Page 110.
    """
    raise NotImplementedError()


def nywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Normalized Yuler-Walker Systems of Equations."""
    raise NotImplementedError()


def ndeterministic(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Normal Determenistic Systems of Equations."""
    raise NotImplementedError()


def back_substitution(x: list[float], p: int) -> NoReturn:
    """Convinient recursion for an all-pole model."""
    raise NotImplementedError()
