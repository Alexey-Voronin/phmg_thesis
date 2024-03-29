from firedrake.preconditioners.pmg import PMGPC, PMGBase
from firedrake import FiniteElement


def get_cell_type(element):
    cell = element.cell
    dim = cell.geometric_dimension()
    if dim == 2:  # Choose between triangle and quadrilateral
        return "triangle" if cell.cellname() == "triangle" else "quadrilateral"
    elif dim == 3:  # Choose between tetrahedron and hexahedron
        return "tetrahedron" if cell.cellname() == "tetrahedron" else "hexahedron"
    else:
        raise ValueError(f"Unsupported geometric dimension: {dim}")


class Direct_Coarsening_THk_TH2(PMGPC):
    """
    PMG class that coarsens ANY Taylor-Hood discretizations
    directly down to Q2/Q1 or P2/P1 discretization.
    """

    def coarsen_element(self, ele):
        U, P = ele.sub_elements
        u_degree = PMGBase.max_degree(U)
        if u_degree <= 2:
            raise ValueError
        else:
            U_new = PMGBase.reconstruct_degree(U, 2)
            P_new = PMGBase.reconstruct_degree(P, 1)

        ME = U_new * P_new

        return ME


class Direct_Coarsening_SVk_TH2(PMGPC):
    def coarsen_element(self, ele):
        U, P = ele.sub_elements
        u_degree = PMGBase.max_degree(U)

        if P.family() == "Lagrange":
            # dg pressure has been changed to cg
            # stop "coarsening"
            raise ValueError
        else:
            U_new = PMGBase.reconstruct_degree(U, 2) if u_degree > 2 else U
            P_new = FiniteElement("Lagrange", get_cell_type(P), 1)

        try:
            ME = U_new * P_new
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

        return ME


#########################################################
# Experimental Coarsening Strategies


class Direct_Coarsening_Pk(PMGPC):
    """
    PMG class that coarsens ANY P_k discretization
    directly to P_2.

    See class for coarsening schedule.
    """

    def coarsen_element(self, U):
        u_degree = PMGBase.max_degree(U)
        if u_degree == 2:
            raise ValueError
        else:
            U_new = PMGBase.reconstruct_degree(U, 2)

        return U_new


class Gradual_Coarsening_Pk(PMGPC):
    """
    PMG class that coarsens ANY P_k discretization
    gradually to P_2.

    See class for coarsening schedule.
    """

    def coarsen_element(self, U):
        u_degree = PMGBase.max_degree(U)

        deg_map = {
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 4,
            7: 4,
            8: 5,
            9: 5,
            10: 5,
        }

        if u_degree == 2:
            raise ValueError
        else:
            u_degree_new = deg_map[u_degree]
            U_new = PMGBase.reconstruct_degree(U, u_degree_new)

        return U_new


class Gradual_Coarsening_THk_TH2(PMGPC):
    """
    PMG class that coarsens ANY Taylor-Hood discretizations
    gradually to Q2/Q1 or P2/P1 discretization.

    See class for coarsening schedule.
    """

    def coarsen_element(self, ele):
        U, P = ele.sub_elements
        u_degree = PMGBase.max_degree(U)

        deg_map = {
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 4,
            7: 4,
            8: 5,
            9: 5,
            10: 5,
        }

        if u_degree == 2:
            raise ValueError
        else:
            u_degree_new = deg_map[u_degree]
            U_new = PMGBase.reconstruct_degree(U, u_degree_new)
            P_new = PMGBase.reconstruct_degree(P, u_degree_new - 1)

        try:
            ME = U_new * P_new
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
        return ME


class Gradual_Coarsening_SVk_THkhat_TH2(PMGPC):
    """Gradually coarsen SV(k, k-1) discretization by switching
    to TH(khat,khat-1) discretization on the intermediate grids.
    """

    def coarsen_element(self, ele):
        U, P = ele.sub_elements
        u_degree = PMGBase.max_degree(U)

        deg_map = {
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 4,
            7: 4,
            8: 5,
            9: 5,
            10: 5,
            #9: 7, # these works slightly beter
            #10: 7,
        }

        if P.family() == "Lagrange":
            if u_degree == 2:
                raise ValueError
            else:
                u_degree_new = deg_map[u_degree]
                U_new = PMGBase.reconstruct_degree(U, u_degree_new)
                P_new = PMGBase.reconstruct_degree(P, u_degree_new - 1)
        else:
            u_degree_new = u_degree
            u_degree_new = deg_map[u_degree]
            U_new = PMGBase.reconstruct_degree(U, u_degree_new)
            P_new = FiniteElement("Lagrange", get_cell_type(P), u_degree_new - 1)

        try:
            ME = U_new * P_new
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
        return ME


class Gradual_Coarsening_SVk_SVkhat_TH2(PMGPC):
    def coarsen_element(self, ele):
        U, P = ele.sub_elements
        deg_map = {
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 4,
            7: 4,
            8: 5,
            9: 7,
            10: 7,
        }

        u_degree = PMGBase.max_degree(U)
        u_degree_new = deg_map[u_degree]

        if P.family() == "Lagrange":
            raise ValueError
        elif u_degree_new == 2:
            U_new = PMGBase.reconstruct_degree(U, u_degree_new)
            P_new = FiniteElement("Lagrange", get_cell_type(P), u_degree_new - 1)

        else:
            U_new = PMGBase.reconstruct_degree(U, u_degree_new)
            P_new = PMGBase.reconstruct_degree(P, u_degree_new - 1)
        try:
            ME = U_new * P_new
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
        return ME
