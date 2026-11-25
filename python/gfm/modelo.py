"""Modelo simbólico (CasADi) idéntico al modelo de Julia (julia/src/modelo.jl).

Estados (22): SG [δ, ω, xg, xr] + por GFM [δ, ω, E, qm, SOC, vrc, e, ϑ, pred].
θ = [H1 H2 D1 D2 Kq1 Kq2 K1 K2].
"""
import casadi as ca
import numpy as np

NG = 9
NX = 4 + 2 * NG
N_PF = 8                      # iteraciones de punto fijo de la carga a potencia constante (igual en Julia)
SALIDAS = ["f_sg", "f1", "f2", "I1sq", "I2sq", "ib1", "ib2", "vbat1", "vbat2", "pb1", "pb2",
           "SOC1", "SOC2", "E1", "E2", "V1sq", "V2sq", "V3sq", "V4sq", "V5sq", "ploss", "p1", "p2",
           "th1", "th2", "pred1", "pred2"]
IDX = {n: i for i, n in enumerate(SALIDAS)}


def igfm(i):
    return 4 + NG * i            # i = 0, 1


def rampa(x, eps):
    """Rampa suave ≈ max(x, 0) de ancho eps (softplus en forma numéricamente estable)."""
    return ca.fmax(x, 0) + eps * ca.log1p(ca.exp(-ca.fabs(x) / eps))


def sigm(z):
    return (1 + ca.tanh(z / 2)) / 2


def _algebra(x, c, e):
    G, B = e["Y"].real, e["Y"].imag
    Er = ca.vertcat(c["Eg"] * ca.cos(x[0]), *[x[igfm(i) + 2] * ca.cos(x[igfm(i)]) for i in range(2)])
    Ei = ca.vertcat(c["Eg"] * ca.sin(x[0]), *[x[igfm(i) + 2] * ca.sin(x[igfm(i)]) for i in range(2)])
    Ir, Ii = ca.mtimes(G, Er) - ca.mtimes(B, Ei), ca.mtimes(G, Ei) + ca.mtimes(B, Er)
    Wr, Wi = e["W"].real, e["W"].imag
    Vr0, Vi0 = ca.mtimes(Wr, Er) - ca.mtimes(Wi, Ei), ca.mtimes(Wr, Ei) + ca.mtimes(Wi, Er)
    Vbr, Vbi = Vr0, Vi0
    if np.any(e["Sc"]):                                         # carga a potencia constante: punto fijo en V
        Zr, Zi, Pc, Qc = e["Z"].real, e["Z"].imag, e["Sc"].real, e["Sc"].imag
        for _ in range(N_PF):
            v2 = Vbr ** 2 + Vbi ** 2
            jr, ji = -(Pc * Vbr + Qc * Vbi) / v2, -(Pc * Vbi - Qc * Vbr) / v2         # -conj(S/V)
            Vbr = Vr0 + ca.mtimes(Zr, jr) - ca.mtimes(Zi, ji)
            Vbi = Vi0 + ca.mtimes(Zr, ji) + ca.mtimes(Zi, jr)
        Ir = Ir - ca.mtimes(Wr.T, jr) + ca.mtimes(Wi.T, ji)
        Ii = Ii - ca.mtimes(Wr.T, ji) - ca.mtimes(Wi.T, jr)
    P = Er * Ir + Ei * Ii
    pg = P[0] * c["Sb"] / c["Sg"]
    gfm = []
    for i in range(2):
        k, f = igfm(i), c["Sb"] / c["S"][i]
        Vr, Vi = Er[i + 1] + c["Xf"][i] * Ii[i + 1], Ei[i + 1] - c["Xf"][i] * Ir[i + 1]
        p = P[i + 1] * f
        q = (Vi * Ir[i + 1] - Vr * Ii[i + 1]) * f
        isq = (Ir[i + 1] ** 2 + Ii[i + 1] ** 2) * f ** 2
        pb = (p + c["Rconv"][i] * isq - e["pfv"][i]) * c["S"][i] / c["Pn"][i]
        vo = c["ocv0"] + c["ocv1"] * x[k + 4] - x[k + 5]
        ib = (vo - ca.sqrt(ca.fmax(vo ** 2 - 4 * c["R0"] * pb, 1e-9))) / (2 * c["R0"])
        ilim = c["Icont"] + (c["Imax"] - c["Icont"]) * sigm((c["mth"] * c["B"] - x[k + 7]) / (c["ath"] * c["B"]))
        gfm.append(dict(p=p, q=q, U=ca.sqrt(Vr ** 2 + Vi ** 2), isq=isq, pb=pb, ib=ib, vbat=vo - c["R0"] * ib,
                        ilim=ilim))
    Vsq = Vbr ** 2 + Vbi ** 2
    ploss = ca.sum1(P) - ca.dot(e["GL"], Vsq) - e["Sc"].real.sum()
    return pg, gfm, Vsq, ploss


def construir(c, e):
    """Funciones CasADi f(x,θ) -> dx/dt y salidas(x,θ) -> vector SALIDAS para el escenario e.
    e["lim"] = 1 incluye el limitador de sobrecarga en la dinámica; 0 lo deja fuera (pred no actúa)."""
    x, th = ca.SX.sym("x", NX), ca.SX.sym("th", 8)
    pg, g, Vsq, ploss = _algebra(x, c, e)
    pm = c["pm0"] + x[3] + c["FH"] * (x[2] - x[3])
    dx = [c["wb"] * (x[1] - 1),
          (pm - pg - c["Dg"] * (x[1] - 1)) / (2 * c["Hg"]),
          (-(x[1] - 1) / c["Rg"] - x[2]) / c["TG"],
          (x[2] - x[3]) / c["TR"]]
    pred_tot = []
    for i in range(2):
        k, gi = igfm(i), g[i]
        H, D, Kq, K = th[i], th[2 + i], th[4 + i], th[6 + i]
        ieff = 0.5 * (1 + c["eta"]) * gi["ib"] + 0.5 * (1 - c["eta"]) * ca.sqrt(gi["ib"] ** 2 + 1e-6)
        iac = ca.sqrt(gi["isq"])
        exceso = rampa(iac - gi["ilim"], c["eps_i"]) + c["Pn"][i] / c["S"][i] * rampa(gi["pb"] - 1, c["eps_i"])
        pred = e["lim"] * (x[k + 8] + c["kp_lim"] * exceso)        # limitador PI
        pred_tot.append(x[k + 8] + c["kp_lim"] * exceso)
        dx += [c["wb"] * (x[k + 1] - 1),
               (c["p0"][i] - pred - gi["p"] - D * (x[k + 1] - 1)) / (2 * H),
               K * (Kq * (c["U0"][i] - gi["U"]) + c["q0"][i] - x[k + 3]),
               (gi["q"] - x[k + 3]) / c["tau_q"][i],
               -ieff / (3600 * c["Eh"][i]),
               (c["R1"] * gi["ib"] - x[k + 5]) / c["tau1"],
               (gi["pb"] - e["pb0"][i]) * c["Pn"][i],
               rampa(gi["isq"] - c["Icont"] ** 2, c["eps_th"]),
               c["klim"] * exceso
               - x[k + 8] / c["tau_lib"] * sigm((gi["ilim"] - iac) / c["eps_i"]) * sigm((1 - gi["pb"]) / c["eps_i"])]
    f0 = c["f0"]
    out = [f0 * x[1], f0 * x[igfm(0) + 1], f0 * x[igfm(1) + 1], g[0]["isq"], g[1]["isq"],
           g[0]["ib"], g[1]["ib"], g[0]["vbat"], g[1]["vbat"], g[0]["pb"], g[1]["pb"],
           x[igfm(0) + 4], x[igfm(1) + 4], x[igfm(0) + 6], x[igfm(1) + 6],
           *[Vsq[b] for b in range(5)], ploss, g[0]["p"], g[1]["p"],
           x[igfm(0) + 7], x[igfm(1) + 7], *pred_tot]
    return (ca.Function("f", [x, th], [ca.vertcat(*dx)]),
            ca.Function("salidas", [x, th], [ca.vertcat(*out)]))


def simular(f, x0, th, t):
    """Simulación con CVODES sobre la malla t (se usa como punto inicial de las formulaciones directas)."""
    x, p = ca.MX.sym("x", NX), ca.MX.sym("p", 8)
    F = ca.integrator("F", "cvodes", {"x": x, "p": p, "ode": f(x, p)}, 0, list(t[1:]),
                      {"abstol": 1e-9, "reltol": 1e-7, "max_num_steps": 100000})
    X = np.asarray(F(x0=x0, p=th)["xf"])
    return np.hstack([np.asarray(x0).reshape(-1, 1), X])


def sensibilidad_tension(c, e):
    """α_i = ∂U_i/∂E_i y β_i = ∂q_i/∂E_i en el punto de operación (δ fijo).
    Linealizando el lazo Q-V: dE/dt ≈ -K (Kq α + β) ΔE  →  τ_v = 1 / (K (Kq α + β))."""
    x = ca.SX.sym("x", NX)
    _, g, _, _ = _algebra(x, c, e)
    a, b = [], []
    for i in range(2):
        Ei = x[igfm(i) + 2]
        fa = ca.Function("a", [x], [ca.jacobian(g[i]["U"], Ei), ca.jacobian(g[i]["q"], Ei)])
        ai, bi = fa(e["x0"])
        a.append(float(ai))
        b.append(float(bi))
    return np.array(a), np.array(b)
