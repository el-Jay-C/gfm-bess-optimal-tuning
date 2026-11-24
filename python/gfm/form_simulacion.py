"""Formulaciones basadas en simulación (caja negra): el modelo de Julia se evalúa como oráculo.

F1 - Réplica del artículo base (Wang et al., 2024): lazo activo por SQP (min nadir s.a. P ≤ Pmax) y luego
     lazo reactivo por PSO (pérdidas + penalización de tensión). Su modelo no tiene limitador ni BESS.
F2 - SQP conjunto (SLSQP, gradientes por diferencias finitas) con las restricciones agregadas en el tiempo.
F3 - PSO conjunto con penalización cuadrática de las restricciones.
ProblemaCN arma el problema para F2 y F3 y admite, además:
  - modo "R" (restricciones que impiden que el limitador actúe) o "L" (limitador en la dinámica);
  - subconjuntos de restricciones (ablación) y pesos del objetivo distintos;
  - varios escenarios con una sola sintonía: "minmax", "esperado" o "cvar";
  - regla de decisión afín θ(ξ) = θ0 + ξ·Δθ sobre una condición ξ observable antes del evento (recurso).
"""
import numpy as np
from scipy.optimize import minimize

from .datos import g_tension, limites, restricciones_estaticas
from .pso import pso

GRUPOS = {"R2": ["df_max", "rocof"],
          "R3": ["I1_max", "I2_max", "th1", "th2", "I1_fin", "I2_fin"],
          "R4": ["ib1_max", "ib2_max", "vbat1_min", "vbat2_min", "pb1_max", "pb2_max",
                 "pb1_sost", "pb2_sost", "SOC1_min", "SOC2_min", "SOC1_max", "SOC2_max"],
          "R5": ["E1_serv", "E2_serv"],
          "R6": ["V_banda"]}
SOSTENIDAS = {"th1", "th2", "I1_fin", "I2_fin", "pb1_max", "pb2_max"}     # en modo L las maneja el limitador
FILAS_EST = {"R7": [0], "R8": [1, 2, 3, 4], "R11": [5, 6]}


class ProblemaCN:
    def __init__(self, orac, p, c, escs, idx, ab, modo="R", grupos=("R2", "R3", "R4", "R5", "R6"),
                 estaticas=("R7", "R8", "R9", "R11"), pesos_J=None, agregacion=None, pesos=None, alpha=0.8, xi=None,
                 reserva=None, p_max=False, eps_V=None, zeta_min=None, margen=0.0):
        """escs: datos de los escenarios (para las restricciones estáticas); idx: sus índices en el oráculo.
        xi: valor de la condición observable de cada escenario (activa la regla afín).
        reserva: fracción de energía reservada (por defecto la del archivo de parámetros).
        p_max: agrega las restricciones del artículo base, P ≤ Pmax del inversor y del BESS.
        eps_V: cota relativa a θ0 por escenario y modo; infinito la desactiva.
        zeta_min: amortiguamiento mínimo en el equilibrio post-evento, con el limitador libre en L; 0 lo desactiva
        (sólo escenarios exportados por Julia, por defecto el del archivo de parámetros).
        margen: exige g_oráculo ≤ -margen en las restricciones de trayectoria seleccionadas."""
        o = p["optimizacion"]
        self.orac, self.p, self.c, self.escs, self.idx, self.ab = orac, p, c, escs, list(idx), ab
        self.lim = 1 if modo == "L" else 0
        self.eps_V = o["eps_V"] if eps_V is None else eps_V
        self.eps = (self.eps_V * np.array([orac.evaluar(o["theta_inicial"], k, self.lim)[0]["vdev"]
                                         for k in self.idx]) if np.isfinite(self.eps_V) else None)
        self.zeta_min, self.margen = o["zeta_min"] if zeta_min is None else zeta_min, margen
        if self.zeta_min > 0:
            assert max(self.idx) < len(p["escenario"]), "ζ necesita un escenario de datos.cargar()"
        self.cache_zeta = {}
        nombres = [n for gname in grupos for n in GRUPOS[gname] if not (modo == "L" and n in SOSTENIDAS)]
        self.cols = [orac.ig[n] for n in nombres]
        self.reserva, self.p_max = reserva, p_max
        self.esc_E = [self.cols.index(orac.ig[n]) for n in ("E1_serv", "E2_serv") if orac.ig[n] in self.cols]
        self.estaticas = estaticas
        self.wJ = (o["wf"], o["wv"], o["wl"]) if pesos_J is None else pesos_J
        self.agregacion = agregacion if len(self.idx) > 1 else None
        self.pesos = np.ones(len(self.idx)) / len(self.idx) if pesos is None else np.asarray(pesos, float)
        self.alpha, self.xi = alpha, xi
        self.lb, self.ub = limites(p)
        self.n_extra = {None: 0, "esperado": 0, "minmax": 1, "cvar": 1 + len(self.idx)}[self.agregacion]
        self.dim = 8 * (2 if xi is not None else 1) + self.n_extra
        self.cotas = [(0, 1)] * 8 + ([(-1, 1)] * 8 if xi is not None else []) + [(0, 3)] * self.n_extra

    # --- variables escaladas -> sintonía de cada escenario
    def theta(self, z, s=0):
        th = self.lb + np.asarray(z[:8]) * (self.ub - self.lb)
        if self.xi is not None:
            th = th + self.xi[s] * np.asarray(z[8:16]) * (self.ub - self.lb)
        return th

    def z_de(self, th):
        th = np.asarray(th, float)
        z = np.zeros(self.dim)
        z[:8] = (th[:8] - self.lb) / (self.ub - self.lb)
        if self.xi is not None and len(th) >= 16:
            z[8:16] = th[8:16] / (self.ub - self.lb)
        if self.n_extra:
            Js = np.array([self.J_esc(z, s) for s in range(len(self.idx))])
            z[self.dim - self.n_extra] = Js.max() if self.agregacion == "minmax" else Js.mean()
            if self.agregacion == "cvar":
                z[self.dim - self.n_extra + 1:] = np.maximum(Js - Js.mean(), 0)
        return z

    def J_esc(self, z, s):
        m = self.orac.evaluar(self.theta(z, s), self.idx[s], self.lim)[0]
        wf, wv, wl = self.wJ
        return wf * m["dfmax_Hz"] / self.p["optimizacion"]["df_ref"] + wv * m["vdev"] + wl * m["perdidas"]

    def J(self, z):
        Js = np.array([self.J_esc(z, s) for s in range(len(self.idx))])
        if self.agregacion is None:
            return Js[0]
        if self.agregacion == "esperado":
            return float(self.pesos @ Js)
        ex = z[self.dim - self.n_extra:]
        if self.agregacion == "minmax":
            return ex[0]
        return ex[0] + float(self.pesos @ ex[1:]) / (1 - self.alpha)

    def J_real(self, z):
        """Valor del criterio agregado sin variables auxiliares (para reportar)."""
        Js = np.array([self.J_esc(z, s) for s in range(len(self.idx))])
        if self.agregacion in (None, "esperado"):
            return float(self.pesos @ Js) if self.agregacion else Js[0]
        if self.agregacion == "minmax":
            return Js.max()
        eta = min(Js, key=lambda e: e + self.pesos @ np.maximum(Js - e, 0) / (1 - self.alpha))
        return eta + self.pesos @ np.maximum(Js - eta, 0) / (1 - self.alpha)

    def zeta(self, z, s=0):
        from .certificado import linealizar
        th = self.theta(z, s)
        clave = (s, *th)
        if clave not in self.cache_zeta:
            try:                                            # sin equilibrio post-evento: se penaliza como inestable
                A, res = linealizar(self.c, {**self.escs[s], "lim": self.lim}, th)
                ev = np.linalg.eigvals(A)
                self.cache_zeta[clave] = float(np.min(-ev.real / np.abs(ev))) if res < 1e-6 else -1.0
            except (RuntimeError, np.linalg.LinAlgError):
                self.cache_zeta[clave] = -1.0
        return self.cache_zeta[clave]

    def g(self, z):
        """Todas las restricciones como g(z) ≥ 0 (convención de SLSQP), normalizadas."""
        partes = []
        r0 = self.p["bess"]["reserva"]
        for s in range(len(self.idx)):
            m, gs = self.orac.evaluar(self.theta(z, s), self.idx[s], self.lim)
            gs = gs[self.cols].copy()
            if self.reserva is not None:                    # g = E_serv/E_res - 1 con otra reserva
                gs[self.esc_E] = (gs[self.esc_E] + 1) * r0 / self.reserva - 1
            partes.append(-gs - self.margen)
            if self.eps is not None:
                partes.append(np.array([1 - m["vdev"] / self.eps[s]]))
            if self.zeta_min > 0:
                partes.append(np.array([(self.zeta(z, s) - self.zeta_min) / self.zeta_min]))
            if self.p_max:
                partes.append(1.0 - np.array([m["p1_max"], m["p2_max"], m["pb1_pico"], m["pb2_pico"]]))
        distintas = range(len(self.idx)) if self.xi is not None else [0]
        for s in distintas:
            th = self.theta(z, s)
            G, h = restricciones_estaticas(self.p, self.c, [self.escs[s]] if self.xi is not None else self.escs,
                                           self.reserva)
            for nombre, filas in FILAS_EST.items():
                if nombre in self.estaticas:
                    partes.append((h[filas] - G[filas] @ th) / np.maximum(np.abs(h[filas]), 1.0))
            if "R9" in self.estaticas:
                partes.append(-g_tension(th, self.ab, self.p, self.c) / 2.2)
            if self.xi is not None:
                partes += [th - self.lb, self.ub - th]
        ex = z[self.dim - self.n_extra:]
        if self.agregacion == "minmax":
            partes.append(ex[0] - np.array([self.J_esc(z, s) for s in range(len(self.idx))]))
        elif self.agregacion == "cvar":
            partes.append(ex[0] + ex[1:] - np.array([self.J_esc(z, s) for s in range(len(self.idx))]))
        return np.concatenate(partes)

    def viol(self, z):
        g = self.g(z)
        g = g[:-len(self.idx)] if self.n_extra else g
        return max(0.0, -g.min())


def f2_sqp(prob, th0, hist=None, maxiter=200, ftol=1e-8, eps=1e-4, z0=None):
    """SQP (SLSQP: subproblema cuadrático con Hessiano BFGS, gradientes por diferencias finitas)."""
    z0 = prob.z_de(th0) if z0 is None else z0
    if prob.n_extra:                                    # variables auxiliares factibles en el punto inicial
        Js = np.array([prob.J_esc(z0, s) for s in range(len(prob.idx))])
        z0[prob.dim - prob.n_extra] = Js.max() if prob.agregacion == "minmax" else Js.mean()
        if prob.agregacion == "cvar":
            z0[prob.dim - prob.n_extra + 1:] = np.maximum(Js - Js.mean(), 0)

    def cb(z):
        if hist is not None:
            hist.registrar(prob.J_real(z), prob.viol(z), prob.theta(z), prob.orac.n_eval)

    cb(z0)
    r = minimize(prob.J, z0, method="SLSQP", bounds=prob.cotas, callback=cb,
                 constraints=[{"type": "ineq", "fun": prob.g}], options={"maxiter": maxiter, "ftol": ftol, "eps": eps})
    return r.x


def f3_pso(prob, hist=None, semilla=0, n_part=24, n_iter=250, rho=1e4, th0=None):
    """PSO con penalización cuadrática exterior de las restricciones."""
    def fit(z):
        return prob.J(z) + rho * np.sum(np.minimum(prob.g(z), 0) ** 2)

    def cb(it, z, fz):
        if hist is not None:
            hist.registrar(prob.J_real(z), prob.viol(z), prob.theta(z), prob.orac.n_eval)

    z, _ = pso(fit, prob.dim, n_part=n_part, n_iter=n_iter, semilla=semilla, callback=cb,
               z_ini=None if th0 is None else prob.z_de(th0))
    return z


def f1_paper(orac, p, c, esc, hist, semilla=0, n_iter_pso=1000, th0=None):
    """Réplica del artículo base. Se diseña con su propio modelo: sin limitador (lim = 0) y sin R3-R9."""
    lb, ub = limites(p)
    th0 = np.array(p["optimizacion"]["theta_inicial"], float) if th0 is None else th0.copy()
    n = p["norma"]
    iA, iB = [0, 1, 2, 3], [4, 5, 6, 7]

    def con_z(z, idx, base):
        th = base.copy()
        th[idx] = lb[idx] + z * (ub - lb)[idx]
        return th

    # Etapa A: SQP sobre (H, D), mínima Δf_max sujeto a P ≤ Pmax del inversor y P ≤ Pfv + P_ESS,max
    def fA(z):
        return orac.evaluar(con_z(z, iA, th0), esc, 0)[0]["dfmax_Hz"]

    def gA(z):
        m = orac.evaluar(con_z(z, iA, th0), esc, 0)[0]
        return 1.0 - np.array([m["p1_max"], m["p2_max"], m["pb1_pico"], m["pb2_pico"]])

    def cbA(z):
        hist.registrar(fA(z), max(0, -gA(z).min()), con_z(z, iA, th0), orac.n_eval)

    zA0 = (th0[iA] - lb[iA]) / (ub - lb)[iA]
    cbA(zA0)
    rA = minimize(fA, zA0, method="SLSQP", bounds=[(0, 1)] * 4, callback=cbA,
                  constraints=[{"type": "ineq", "fun": gA}], options={"maxiter": 100, "ftol": 1e-8, "eps": 1e-4})
    thA = con_z(rA.x, iA, th0)

    # Etapa B: PSO sobre (Kq, K) con la aptitud del artículo: pérdidas + λ·desviación de tensión
    def fB(z):
        m = orac.evaluar(con_z(z, iB, thA), esc, 0)[0]
        fuera = m["V_min"] < n["V_min_normal"] or m["V_max"] > n["V_max_normal"]
        return m["perdidas"] + (100.0 if fuera else 0.0) * m["vdev_paper"]

    def cbB(it, z, fz):
        hist.registrar(fz, 0.0, con_z(z, iB, thA), orac.n_eval)

    zB, _ = pso(fB, 4, n_part=12, n_iter=n_iter_pso, semilla=semilla, callback=cbB)
    return con_z(zB, iB, thA)
