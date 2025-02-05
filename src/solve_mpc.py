from src.control.mpc import MPC, MPCParams


def solve_mpc(
    mpc_params: MPCParams,
    solver: str,
    **kwargs,
):
    mpc_qp = MPC(mpc_params)
    qp_sol = mpc_qp.qp_solve(mpc_qp.problem, solver=solver, **kwargs)
    return qp_sol
