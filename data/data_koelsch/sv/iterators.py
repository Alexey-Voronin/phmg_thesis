import sys,os
sys.path.append(os.path.abspath("../../../../../problems/"))
from ldc2d import TwoDimLidDrivenCavityProblem
from ldc3d import ThreeDimLidDrivenCavityProblem
from fpc2d import TwoDimFlowPastCylinderProblem
from pcf3d import ThreeDimPinchedChannelFlowProblem
from bfs2d import TwoDimBackwardFacingStepProblem
from bfs3d import ThreeDimBackwardFacingStepProblem

def ldc2d_iterator(k, diagonal, quadrilateral, solver='phmg'):

    base = 80
    base = int(base/k)
    # 10x10 coarse meshes give weird reults..
    base = base+1 if 2*base == 10 else base
    # Mesh Refinement Details:
    # - ALFI: Uses barycentrically refined coarse-grids in a quadrisection series (4h, 2h, h).
    # - phMG: Employs quadrisection refinement for all coarse-grids, with barycentric refinement
    #   only at the fine-grid level.
    # - For equivalent fine-grid meshes after n refinements, the phMG solver must start
    #   on a grid that is 2*base compared to ALFI.
    if solver in ['phmg', 'afbf']:
        base = base*2
    return TwoDimLidDrivenCavityProblem(base,
                                       diagonal,
                                       quadrilateral)

def ldc3d_iterator(k):
    baseN = 11
    return ThreeDimLidDrivenCavityProblem(max(2,int(baseN/(k))))

def fpc2d_iterator(k):
    return TwoDimFlowPastCylinderProblem(1) #int(max(0, 2-(k-2)/1.35)))

def bfs2d_iterator(k, ref_mesh=False):
    return TwoDimBackwardFacingStepProblem(int(max(0, round(7/1.12**(k-2)))),
                                           ref_mesh=ref_mesh)

def bfs3d_iterator(k):
    return ThreeDimBackwardFacingStepProblem(int(max(0, 4-(k-2)/1.35)))

def pcf3d_iterator(k):
    return ThreeDimPinchedChannelFlowProblem(2) #int(max(0, 4-(k-2)/2)))
