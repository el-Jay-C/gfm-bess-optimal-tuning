"""Certificado de estabilidad de pequeña señal por SDP (LMI de Lyapunov).

Alcance: estabilidad local (pequeña señal) del equilibrio post-perturbación del subsistema AC. No certifica la
trayectoria completa ni la región de atracción. Para una sola matriz A el SDP reproduce lo que entregan los
autovalores; su aporte propio es la función de Lyapunov común a varios escenarios (tasa_certificada_comun).

Se linealiza el subsistema AC (SG + lazos P-f y Q-V de los GFM, ángulos relativos a la máquina) en el
equilibrio post-perturbación, ż = A z. Para una tasa de decaimiento α se busca P tal que
    P ⪰ I,   AᵀP + PA + 2αP ⪯ 0,
que es una LMI (SDP) en P. El mayor α factible se obtiene por bisección (problema cuasiconvexo) y
certifica que todas las trayectorias linealizadas decaen al menos como e^{-αt}. Además, con la región
LMI cónica de Chilali y Gahinet (1996) se certifica un amortiguamiento mínimo ζ de todos los modos:
    X ≻ 0,  [[sinϑ(AX+XAᵀ), cosϑ(AX−XAᵀ)], [cosϑ(XAᵀ−AX), sinϑ(AX+XAᵀ)]] ≺ 0,  ζ = cos ϑ.
Los estados del BESS (SOC, vrc, e) no realimentan la dinámica AC y se excluyen.
"""
import casadi as ca
import cvxpy as cp
import numpy as np

from .modelo import NX, construir, igfm, simular

AC_RED = [1, 2, 3] + [igfm(i) + k for i in range(2) for k in (0, 1, 2, 3)]   # ωg, xg, xr, δi, ωi, Ei, qmi


_CACHE = {}


def _funciones(c, e, T_eq):
    """Simulador, campo reducido y jacobiano del escenario, compilados una vez (la sintonía es parámetro)."""
    AC = AC_RED + ([igfm(i) + 8 for i in range(2)] if e["lim"] else [])      # en L el limitador queda en el lazo
    clave = (id(c), e["lim"], T_eq, e["Y"].tobytes(), e["x0"].tobytes(), e["pfv"].tobytes(), e["Sc"].tobytes())
    if clave in _CACHE:
        return _CACHE[clave]
    f, _ = construir(c, e)
    z, xb, th, wb = ca.SX.sym("z", len(AC)), ca.SX.sym("xb", NX), ca.SX.sym("th", 8), c["wb"]
    x = ca.SX(xb)
    x[0] = 0
    for j, k in enumerate(AC):
        x[k] = z[j]
    dx = f(x, th)
    dz = []
    for j, k in enumerate(AC):
        if k in (igfm(0), igfm(1)):                          # δi relativo: d(δi-δg)/dt = ωb(ωi - ωg)
            dz.append(wb * (z[j + 1] - z[0]))
        else:
            dz.append(dx[k])
    dz = ca.vertcat(*dz)
    pr = ca.vertcat(xb, th)
    F = ca.Function("F", [z, pr], [dz])
    xs, ps = ca.MX.sym("x", NX), ca.MX.sym("p", 8)
    sim = ca.integrator("sim", "cvodes", {"x": xs, "p": ps, "ode": f(xs, ps)}, 0, T_eq,
                        {"abstol": 1e-9, "reltol": 1e-7, "max_num_steps": 100000})
    _CACHE[clave] = (AC, sim, F, ca.rootfinder("eq", "newton", F), ca.Function("J", [z, pr], [ca.jacobian(dz, z)]))
    return _CACHE[clave]


def linealizar(c, e, th, T_eq=60.0):
    """Matriz A del subsistema AC en el equilibrio post-perturbación (ángulos relativos a δg)."""
    AC, sim, F, raiz, J = _funciones(c, e, T_eq)
    xb = np.asarray(sim(x0=e["x0"], p=th)["xf"]).ravel()             # punto de partida cercano al equilibrio
    pr, z0 = np.r_[xb, th], xb[AC].copy()
    for j, k in enumerate(AC):
        if k in (igfm(0), igfm(1)):
            z0[j] = xb[k] - xb[0]
    zeq = np.asarray(raiz(z0, pr)).ravel()
    return np.asarray(J(zeq, pr)), np.max(np.abs(np.asarray(F(zeq, pr))))


def _resolver(pr):
    """Clarabel (punto interior); si falla por mal condicionamiento cerca de la frontera, SCS."""
    for solver, opts in ((cp.CLARABEL, {}), (cp.SCS, {"eps": 1e-7, "max_iters": 50000})):
        try:
            pr.solve(solver=solver, **opts)
            if pr.status == cp.OPTIMAL:
                return True
        except cp.error.SolverError:
            continue
    return False


def lmi_factible(A, alpha):
    n = A.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    M = A.T @ P + P @ A + 2 * alpha * P
    pr = cp.Problem(cp.Minimize(0), [P >> np.eye(n), (M + M.T) / 2 << 0])
    return _resolver(pr)


def _escalar(A):
    """Escalamiento diagonal (similaridad, no cambia los polos) para mejorar el condicionamiento del SDP."""
    s = np.sqrt(np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(A, axis=0), 1e-12)) ** -0.5
    return np.diag(1 / s) @ A @ np.diag(s)


def sector_factible(A, zeta):
    n, th = A.shape[0], np.arccos(zeta)
    X = cp.Variable((n, n), symmetric=True)
    S1, S2 = A @ X + X @ A.T, A @ X - X @ A.T
    M = cp.bmat([[np.sin(th) * S1, np.cos(th) * S2], [-np.cos(th) * S2, np.sin(th) * S1]])
    pr = cp.Problem(cp.Minimize(0), [X >> np.eye(n), (M + M.T) / 2 << 0])
    return _resolver(pr)


def amortiguamiento_certificado(A, tol=1e-3):
    """Mayor ζ tal que todos los polos están en el sector cónico (bisección sobre la región LMI)."""
    As = _escalar(A)
    if not sector_factible(As, 1e-3):
        return -np.inf
    lo, hi = 1e-3, 0.999
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sector_factible(As, mid) else (lo, mid)
    return lo


def tasa_certificada(A, tol=1e-3):
    """Mayor α con LMI factible (bisección). Devuelve -inf si no se certifica estabilidad."""
    As = _escalar(A)
    if not lmi_factible(As, 0.0):
        return -np.inf
    lo, hi = 0.0, max(1.0, -np.min(np.linalg.eigvals(A).real))
    while hi - lo > tol * max(1.0, lo):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lmi_factible(As, mid) else (lo, mid)
    return lo


def lmi_comun_factible(As, alpha):
    n = As[0].shape[0]
    P = cp.Variable((n, n), symmetric=True)
    res = [P >> np.eye(n)]
    for A in As:
        M = A.T @ P + P @ A + 2 * alpha * P
        res.append((M + M.T) / 2 << 0)
    return _resolver(cp.Problem(cp.Minimize(0), res))


def inexistencia_lmi_comun(As):
    """Alternativa dual: Xs ⪰ 0 y Σ(As Xs + Xs Asᵀ) ≻ 0 excluyen cualquier P ≻ 0 común.
    La traza fija la escala; se verifican los residuos antes de aceptar la prueba numérica."""
    from scipy.linalg import matrix_balance
    _, T = matrix_balance(As[0], permute=False)
    Ae = [np.linalg.solve(T, A @ T) for A in As]
    n = len(T)
    Xs = [cp.Variable((n, n), symmetric=True) for _ in Ae]
    t = cp.Variable()
    M = sum(A @ X + X @ A.T for A, X in zip(Ae, Xs))
    pr = cp.Problem(cp.Maximize(t), [X >> 0 for X in Xs] +
                    [sum(cp.trace(X) for X in Xs) == 1, M >> t * np.eye(n)])
    pr.solve(solver=cp.CLARABEL, tol_feas=1e-10, tol_gap_abs=1e-10, tol_gap_rel=1e-10, max_iter=500)
    return (pr.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and
            min(np.linalg.eigvalsh(X.value).min() for X in Xs) >= -1e-9 and
            np.linalg.eigvalsh(M.value).min() > 1e-8)


def tasa_certificada_comun(As, tol=1e-3):
    """Mayor α con una única P ≻ 0 tal que AₛᵀP + PAₛ + 2αP ⪯ 0 para todas las matrices Aₛ (bisección).
    Es una función de Lyapunov cuadrática común: certifica la tasa α para cada escenario y para cualquier
    combinación convexa de ellos (incertidumbre politópica), algo que los autovalores por separado no entregan.
    Devuelve NaN si la alternativa dual confirma que no existe P común para α ≥ 0."""
    A0 = As[0]
    s = np.sqrt(np.maximum(np.linalg.norm(A0, axis=1) * np.linalg.norm(A0, axis=0), 1e-12)) ** -0.5
    Ae = [np.diag(1 / s) @ A @ np.diag(s) for A in As]            # misma similaridad para todas
    if not lmi_comun_factible(Ae, 0.0):
        assert inexistencia_lmi_comun(As), "Fallo numérico del SDP común: sin certificado dual de inexistencia"
        return np.nan
    lo, hi = 0.0, min(-np.max(np.linalg.eigvals(A).real) for A in As)
    while hi - lo > tol * max(1.0, lo):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lmi_comun_factible(Ae, mid) else (lo, mid)
    return lo


if __name__ == "__main__":     # verificación: sistema con autovalores conocidos
    A = np.array([[-1.0, 10.0], [-10.0, -1.0]])             # λ = -1 ± 10j  →  α* = 1
    a = tasa_certificada(A)
    assert abs(a - 1.0) < 1e-2, a
    assert tasa_certificada(np.array([[0.5, 0.0], [0.0, -1.0]])) == -np.inf
    z = amortiguamiento_certificado(A)                       # ζ = 1/√101 ≈ 0,0995
    assert abs(z - 1 / np.sqrt(101)) < 2e-3, z
    B = np.array([[-2.0, 5.0], [-5.0, -2.0]])                # λ = -2 ± 5j
    ac = tasa_certificada_comun([A, B])
    assert 0.5 < ac <= 1.0 + 1e-2, ac                        # no puede superar la tasa del más lento
    print("Certificado LMI OK, α* =", a, " α común =", ac)
