import sys,os
sys.path.append(os.path.abspath("../../../../../problems/"))
from ldc2d import TwoDimLidDrivenCavityProblem
from ldc3d import ThreeDimLidDrivenCavityProblem
from fpc2d import TwoDimFlowPastCylinderProblem
from pcf3d import ThreeDimPinchedChannelFlowProblem
from bfs2d import TwoDimBackwardFacingStepProblem
from bfs3d import ThreeDimBackwardFacingStepProblem

def ldc2d_iterator(k, diagonal, quadrilateral):
    return TwoDimLidDrivenCavityProblem(round(100/(k/1.5))+1,
                                           diagonal,
                                           quadrilateral)

def ldc3d_iterator(k):
    baseN = 13
    return ThreeDimLidDrivenCavityProblem(max(3,int(baseN/1.5**(k-2))))

def fpc2d_iterator(k):
    return TwoDimFlowPastCylinderProblem(int(max(0, 8-(k-2)/1.35)))

def bfs2d_iterator(k):
    return TwoDimBackwardFacingStepProblem(int(max(0, 9-(k-2)/1.35)))

def bfs3d_iterator(k):
    return ThreeDimBackwardFacingStepProblem(int(max(0, 4-(k-2)/1.35)))

def pcf3d_iterator(k):
    return ThreeDimPinchedChannelFlowProblem(2) #int(max(0, 4-(k-2)/2)))
