
from qe.fundamental_types import (
    Tuple,
    Determinant,
    Energy,
    List,
)
from collections import defaultdict
from qe.integral_indexing_utils import compound_idx4
import math
import numpy as np

def sorted_wf(psi_coef: List[float], psi_det: List[Determinant]) -> Tuple[List[float], List[Determinant]]:
    """
    return copy of psi_coef and psi_det, sorted by decreasing magnitude of psi_coef
    """
    psi_coef_sorted = []
    psi_det_sorted = []
    for i in np.argsort(-np.abs(psi_coef)):
        psi_coef_sorted.append(psi_coef[i])
        psi_det_sorted.append(psi_det[i])
    return psi_coef_sorted, psi_det_sorted

def sorted_wf_thresh(psi_coef: List[float], psi_det: List[Determinant], thresholds: List[float]=[]) -> Tuple[List[float], List[Determinant], List[int]]:
    """
    return copy of psi_coef and psi_det, sorted by decreasing magnitude of psi_coef
    also return list of det indices j such that \sum_{i=1}^{j[k]} |c_i|^2 <= thresholds[k]
    """
    psi_coef_sorted = []
    psi_det_sorted = []
    tot = 0.0
    ndet_thresh = [-1]*len(thresholds)
    for i in np.argsort(-np.abs(psi_coef)):
        psi_coef_sorted.append(psi_coef[i])
        psi_det_sorted.append(psi_det[i])
        #TODO: just do this after the loop with np cumsum and find
        tot += np.abs(psi_coef[i])**2
        for j,(tj,nj) in enumerate(zip(thresholds,ndet_thresh)):
            if (nj < 0) and (tj < tot):
                ndet_thresh[j] = len(psi_coef_sorted)
    assert all(ni >= 0 for ni in ndet_thresh)
    return psi_coef_sorted, psi_det_sorted, ndet_thresh

def ndet_thresh_from_sorted_coef(psi_coef: List[float], thresholds: List[float]=[]) -> List[int]:
    """
    psi_coef: coefs ordered by decreasing magnitude
    thresholds: list of floats in range [0,1]
    return: list of ints n_j (one for each threshold) such that sum_{i=1}^{n_j} |c[i]|^2 <= thresholds[j]
    """
    cumabs = np.cumsum(np.square(psi_coef,dtype="float"))
    assert np.all(np.diff(cumabs,n=2) <= 0)
    return np.searchsorted(cumabs,thresholds).tolist()

