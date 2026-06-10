def test_pfresult_branch_losses_derived():
    """loss = S_from + S_to (производные свойства, МВт/МВАр)."""
    import numpy as np

    from gridpf.contract.types import PFResult

    r = PFResult(
        converged=True,
        iterations_gs=0,
        iterations_nr=1,
        V=np.ones(2, dtype=np.complex128),
        bus_ids=np.array([1, 2]),
        S_from=np.array([10.0 + 5.0j, -3.0 + 1.0j]),
        S_to=np.array([-9.5 - 4.0j, 3.2 - 1.4j]),
    )
    np.testing.assert_allclose(r.branch_loss_p, [0.5, 0.2])
    np.testing.assert_allclose(r.branch_loss_q, [1.0, -0.4])
