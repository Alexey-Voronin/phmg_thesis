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

def ldc3d_iterator(k, hexahedral, ref=1):
    
    if int(ref) == 1:
        NE = max(5, int(20/1.25**(k-3)))
    else:
        NE = max(4, int(14/1.2**(k-3)))

    return ThreeDimLidDrivenCavityProblem(NE, hexahedral)

def fpc2d_iterator(k):
    return TwoDimFlowPastCylinderProblem(int(max(0, 4-(k-2)/1.35)))

def bfs2d_iterator(k):
    return TwoDimBackwardFacingStepProblem(int(max(0, 7-(k-2)/1.35)))

def bfs3d_iterator(k):
    return ThreeDimBackwardFacingStepProblem(int(max(0, 4-(k-2)/1.35)))

def pcf3d_iterator(k):
    return ThreeDimPinchedChannelFlowProblem(2) #int(max(0, 4-(k-2)/2)))
