from __future__ import print_function, absolute_import
import losses
from losses.SoftmaxNeigLoss import SoftmaxNeigLoss
from losses.KNNSoftmax import KNNSoftmax
from losses.NeighbourLoss import NeighbourLoss
from losses.triplet import TripletLoss
from losses.CenterTriplet import CenterTripletLoss
from losses.GaussianMetric import GaussianMetricLoss
from losses.HistogramLoss import HistogramLoss
from losses.BatchAll import BatchAllLoss
from losses.NeighbourLoss import NeighbourLoss
from losses.DistanceMatchLoss import DistanceMatchLoss
from losses.NeighbourHardLoss import NeighbourHardLoss
from losses.DistWeightLoss import DistWeightLoss
from losses.BinDevianceLoss import BinDevianceLoss
from losses.BinBranchLoss import BinBranchLoss
from losses.MarginDevianceLoss import MarginDevianceLoss
from losses.MarginPositiveLoss import MarginPositiveLoss
from losses.ContrastiveLoss import ContrastiveLoss
from losses.DistWeightContrastiveLoss import DistWeightContrastiveLoss
from losses.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from losses.DistWeightDevBranchLoss import DistWeightDevBranchLoss
from losses.DistWeightNeighbourLoss import DistWeightNeighbourLoss
from losses.BDWNeighbourLoss import BDWNeighbourLoss
from losses.EnsembleDWNeighbourLoss import EnsembleDWNeighbourLoss
from losses.BranchKNNSoftmax import BranchKNNSoftmax


__factory = {
    'softneig': SoftmaxNeigLoss,
    'knnsoftmax': KNNSoftmax,
    'neighbour': NeighbourLoss,
    'triplet': TripletLoss,
    'histogram': HistogramLoss,
    'gaussian': GaussianMetricLoss,
    'batchall': BatchAllLoss,
    'neighard': NeighbourHardLoss,
    'bin': BinDevianceLoss,
    'binbranch': BinBranchLoss,
    'margin': MarginDevianceLoss,
    'positive': MarginPositiveLoss,
    'con': ContrastiveLoss,
    'distweight': DistWeightLoss,
    'distance_match': DistanceMatchLoss,
    'dwcon': DistWeightContrastiveLoss,
    'dwdev': DistWeightBinDevianceLoss,
    'dwneig': DistWeightNeighbourLoss,
    'dwdevbranch': DistWeightDevBranchLoss,
    'bdwneig': BDWNeighbourLoss,
    'edwneig': EnsembleDWNeighbourLoss,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)



def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]( *args, **kwargs)