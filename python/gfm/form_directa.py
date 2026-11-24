"""Formulaciones de optimización dinámica directa (CasADi + IPOPT, derivadas exactas por AD).

F4 - Transcripción directa por colocación trapezoidal: estados en todos los nodos como variables.
F5 - Disparo múltiple: estados sólo en los nodos de cada intervalo; RK4 dentro de cada intervalo.
Multiescenario sobre F4 (una sola sintonía θ para todos los escenarios, sin decisiones de recurso):
    "minmax"   min_θ max_s J_s (epígrafe),   "esperado"  min_θ Σ π_s J_s,
    "cvar"     min_{θ,η} η + 1/(1-α) Σ π_s max(J_s - η, 0).
Dos modelos de los límites sostenidos (corriente nominal y potencia nominal del BESS):
    modo "R": restricciones que impiden que el limitador actúe (pb ≤ 1 en todo t, i(T) ≤ I_cont, ϑ(T) ≤ B);
              la dinámica no incluye el limitador.
    modo "L": el limitador está en la dinámica y esas restricciones se omiten.
En ambos modos se imponen en cada nodo los límites de pico (corriente máxima, corriente y tensión del banco,
SOC), la banda de tensión, el RoCoF en ventana y el nadir mediante la variable epígrafe z_f ≥ |f(t) - f0|.
"""
import time

import casadi as ca
import numpy as np

from .datos import cotas_tension, limites, malla, restricciones_estaticas
from .modelo import IDX, NX, construir, sensibilidad_tension, simular


class _Registro(ca.Callback):
    """Callback de iteración de IPOPT: guarda tiempo, objetivo, violación primal y θ."""

    def __init__(self, nw, ng, lbg, ubg, hist):
        ca.Callback.__init__(self)
        self.nw, self.ng, self.lbg, self.ubg, self.hist = nw, ng, lbg, ubg, hist
        self.construct("registro", {})

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == "f":
            return ca.Sparsity.scalar()
        if n in ("x", "lam_x"):
            return ca.Sparsity.dense(self.nw)
        if n in ("g", "lam_g"):
            return ca.Sparsity.dense(self.ng)
        return ca.Sparsity(0, 0)

    def eval(self, arg):
        d = dict(zip([ca.nlpsol_out(i) for i in range(ca.nlpsol_n_out())], arg))
        gv = np.asarray(d["g"]).ravel()
        viol = max(0.0, np.max(self.lbg - gv), np.max(gv - self.ubg))
        if self.hist is not None:
            self.hist.registrar(float(d["f"]), viol, np.asarray(d["x"]).ravel()[:8])
        return [0]


def _rk4(f, M):
    """Función de intervalo de disparo múltiple: M pasos RK4 de tamaño h → estados en cada subpaso (NX × M)."""
    x, th, h = ca.SX.sym("x", NX), ca.SX.sym("th", 8), ca.SX.sym("h")
    xs, xk = [], x
    for _ in range(M):
        k1 = f(xk, th)
        k2 = f(xk + h / 2 * k1, th)
        k3 = f(xk + h / 2 * k2, th)
        k4 = f(xk + h * k3, th)
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(xk)
    return ca.Function("intervalo", [x, th, h], [ca.horzcat(*xs)])


class ProblemaDirecto:
    """Construye una vez el NLP (CasADi + IPOPT) y lo resuelve desde distintos puntos iniciales.
    La reserva de energía y eps_V sólo cambian cotas de restricciones; sus barridos reutilizan el solver."""

    def __init__(self, p, c, escs, metodo="trapecio", modo="R", hess="exact", M=10, max_iter=1000, tol=1e-6,
                 agregacion=None, pesos=None, alpha=0.8, factor=1, pesos_J=None, eps_V=None):
        o, n, b = p["optimizacion"], p["norma"], p["bess"]
        f0 = c["f0"]
        t = malla(p, factor)
        dt = np.diff(t)
        N, T = len(dt), t[-1]
        escs = [{**e, "lim": 1.0 if modo == "L" else 0.0} for e in escs]
        self.p, self.c, self.escs, self.metodo, self.modo, self.M, self.t = p, c, escs, metodo, modo, M, t
        self.eps_V = o["eps_V"] if eps_V is None else eps_V
        self.vdev0 = []
        self.agregacion = agregacion if len(escs) > 1 else None
        wf, wv, wl = (o["wf"], o["wv"], o["wl"]) if pesos_J is None else pesos_J
        lb, ub = limites(p)
        th = ca.MX.sym("th", 8)
        w, lbw, ubw = [th], list(lb), list(ub)
        g, lbg, ubg = [], [], []
        self.filas = {}
        ng = 0

        def agregar(expr, lo, hi, etiqueta=None):
            nonlocal ng
            expr = ca.vec(expr)
            k = expr.numel()
            g.append(expr)
            lbg.extend(np.broadcast_to(lo, k))
            ubg.extend(np.broadcast_to(hi, k))
            if etiqueta is not None:
                self.filas.setdefault(etiqueta, []).extend(range(ng, ng + k))
            ng += k

        vent = n["ventana_rocof"]
        jprev = [max(int(np.searchsorted(t, tk - vent + 1e-9, side="right")) - 1, 0) for tk in t[1:]]
        self.fidx = fidx = [IDX["f_sg"], IDX["f1"], IDX["f2"]]
        wtrap = np.concatenate([[dt[0] / 2], (dt[:-1] + dt[1:]) / 2, [dt[-1] / 2]])
        Ts = b["T_servicio"]
        self.modelos, Js = [], []
        for e in escs:
            f, sal = construir(c, e)
            self.modelos.append((f, sal))
            x0 = ca.DM(e["x0"])
            if metodo == "trapecio":
                Xv = ca.MX.sym("X", NX, N)
                X = ca.horzcat(x0, Xv)
                F = f.map(N + 1)(X, th)
                agregar(X[:, 1:] - X[:, :-1] - (F[:, :-1] + F[:, 1:]) * ca.DM(np.tile(dt / 2, (NX, 1))), 0, 0)
            else:                                           # disparo múltiple
                NI = N // M
                assert N % M == 0 and np.allclose(dt.reshape(NI, M), dt[::M, None]), "malla incompatible con M"
                Xv = ca.MX.sym("X", NX, NI)
                Sub = _rk4(f, M).map(NI)(ca.horzcat(x0, Xv[:, :-1]), th, ca.DM(dt[::M]).T)
                agregar(Sub[:, M - 1::M] - Xv, 0, 0)
                X = ca.horzcat(x0, Sub)
            w.append(ca.vec(Xv))
            lbw += [-np.inf] * Xv.numel()
            ubw += [np.inf] * Xv.numel()

            O = sal.map(N + 1)(X, th)
            zf = ca.MX.sym("zf")
            w.append(zf)
            lbw.append(0.0)
            ubw.append(f0 - n["f_min_perm"])
            Fq = O[fidx, :] - f0
            agregar(Fq - zf, -np.inf, 0)
            agregar(-Fq - zf, -np.inf, 0)
            rw = n["rocof_max"] * vent
            agregar(O[fidx, 1:] - O[fidx, jprev], -rw, rw)
            for i in range(2):
                Isq = O[IDX[f"I{i+1}sq"], :]
                pb = O[IDX[f"pb{i+1}"], :]
                agregar(Isq, -np.inf, c["Imax"] ** 2, "Imax")
                agregar(O[IDX[f"ib{i+1}"], :], -b["i_max"], b["i_max"], "ib")
                agregar(O[IDX[f"vbat{i+1}"], :], b["v_min"], np.inf)
                agregar(O[IDX[f"SOC{i+1}"], :], b["SOC_min"], b["SOC_max"])
                Es = O[IDX[f"E{i+1}"], N] + (Ts - T) * (pb[N] - e["pb0"][i]) * c["Pn"][i]
                agregar(Es, -np.inf, c["Eres"][i], f"E{i}")
                if modo == "R":
                    agregar(pb, -1.0, 1.0, f"pb{i}")
                    agregar(O[IDX[f"th{i+1}"], N], -np.inf, c["B"], f"th{i}")
                    agregar(Isq[N], -np.inf, c["Icont"] ** 2, f"If{i}")
                else:
                    agregar(pb, -1.0, np.inf)
            Vsq = O[[IDX[f"V{k}sq"] for k in range(1, 6)], :]
            agregar(Vsq, n["V_min_normal"] ** 2, n["V_max_normal"] ** 2)

            dV = (ca.sqrt(Vsq) - ca.repmat(ca.DM(c["Vpre"]), 1, N + 1)) / o["dV_ref"]
            vdev = ca.mtimes(ca.sum1(dV ** 2), wtrap) / (5 * T)
            X0 = simular(f, e["x0"], o["theta_inicial"], t)
            O0 = np.asarray(sal.map(N + 1)(X0, o["theta_inicial"]))
            V0 = np.sqrt(O0[[IDX[f"V{k}sq"] for k in range(1, 6)]])
            self.vdev0.append(float(np.sum(((V0 - c["Vpre"][:, None]) / o["dV_ref"]) ** 2, axis=0)
                                    @ wtrap / (5 * T)))
            agregar(vdev, -np.inf, self.eps_V * self.vdev0[-1], "V")
            perd = ca.mtimes(O[IDX["ploss"], :] - c["ploss0"], wtrap) / (T * e["dP"])
            Js.append(wf * zf / o["df_ref"] + wv * vdev + wl * perd)

        G, hst = restricciones_estaticas(p, c, escs)
        agregar(ca.mtimes(ca.DM(G), th), -np.inf, hst, "est")
        a, bb = sensibilidad_tension(c, escs[0])
        agregar(ca.vertcat(*[th[6 + i] * (th[4 + i] * a[i] + bb[i]) for i in range(2)]), *cotas_tension(p, c), "R9")
        self.n_extra = 0
        if self.agregacion is None:
            obj = Js[0]
        elif self.agregacion == "esperado":
            obj = sum(pi * Jk for pi, Jk in zip(pesos, Js))
        elif self.agregacion == "minmax":                   # epígrafe sobre escenarios
            tJ = ca.MX.sym("tJ")
            w.append(tJ); lbw.append(-np.inf); ubw.append(np.inf)
            for Jk in Js:
                agregar(Jk - tJ, -np.inf, 0)
            obj, self.n_extra = tJ, 1
        else:                                               # CVaR_α con variables de cola z_s
            eta, zs = ca.MX.sym("eta"), ca.MX.sym("zs", len(Js))
            w += [eta, zs]; lbw += [-np.inf] + [0.0] * len(Js); ubw += [np.inf] * (1 + len(Js))
            for k, Jk in enumerate(Js):
                agregar(Jk - eta - zs[k], -np.inf, 0)
            obj, self.n_extra = eta + sum(pi * zs[k] for k, pi in enumerate(pesos)) / (1 - alpha), 1 + len(Js)

        self.lbw, self.ubw = np.array(lbw), np.array(ubw)
        self.lbg, self.ubg = np.array(lbg), np.array(ubg)
        nlp = {"x": ca.vertcat(*w), "f": obj, "g": ca.vertcat(*g)}
        self.nw, self.ng = nlp["x"].numel(), ng
        self.cb = _Registro(self.nw, ng, self.lbg, self.ubg, None)
        # disparo múltiple: la función de intervalo ya es SX; expandir el map completo agota la memoria;
        # en multiescenario (len(escs) > 1) la expansión tarda ~20 min en 1 hilo, por lo que se mantiene MX.
        opts = {"expand": metodo == "trapecio" and len(escs) == 1, "print_time": False, "iteration_callback": self.cb,
                "ipopt": {"print_level": 0, "tol": tol, "max_iter": max_iter, "max_cpu_time": 300.0,
                          "hessian_approximation": "exact" if hess == "exact" else "limited-memory", "sb": "yes"}}
        t0 = time.perf_counter()
        self.solver = ca.nlpsol("S", "ipopt", nlp, opts)
        self.t_construccion = time.perf_counter() - t0

    def _inicial(self, th0):
        """Punto inicial: simulación (CVODES) con θ0 en cada escenario."""
        w0 = list(th0)
        for e, (f, sal) in zip(self.escs, self.modelos):
            Xg = simular(f, e["x0"], th0, self.t)
            Og = np.asarray(sal.map(Xg.shape[1])(Xg, th0))
            w0 += list((Xg[:, 1:] if self.metodo == "trapecio" else Xg[:, self.M::self.M]).ravel(order="F"))
            w0.append(float(np.max(np.abs(Og[self.fidx] - self.c["f0"]))))
        return np.array(w0 + [0.6] * self.n_extra)

    def resolver(self, th0, hist=None, reserva=None, eps_V=None):
        lbg, ubg = self.lbg, self.ubg.copy()
        if eps_V is not None:
            ubg[self.filas["V"]] = eps_V * np.array(self.vdev0)
        if reserva is not None:
            hst = restricciones_estaticas(self.p, self.c, self.escs, reserva)[1]
            for i in range(2):
                ubg[self.filas[f"E{i}"]] = hst[1 + i]
            ubg[self.filas["est"]] = hst
        self.cb.lbg, self.cb.ubg, self.cb.hist = lbg, ubg, hist
        t0 = time.perf_counter()
        w0 = self._inicial(np.asarray(th0, float))
        t1 = time.perf_counter()
        sol = self.solver(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=lbg, ubg=ubg)
        t_s = time.perf_counter() - t1
        st = self.solver.stats()
        lam = np.asarray(sol["lam_g"]).ravel()
        r = {"th": np.asarray(sol["x"]).ravel()[:8], "J": float(sol["f"]), "estado": st["return_status"],
             "exito": bool(st["success"]), "iter": st["iter_count"], "t_s": t_s, "t_inicial_s": t1 - t0,
             "t_construccion_s": self.t_construccion, "n_var": self.nw, "n_res": self.ng}
        gv = np.asarray(sol["g"]).ravel()
        r["vdev"] = gv[self.filas["V"]].tolist()
        r["eps"] = ubg[self.filas["V"]].tolist()
        for k, filas in self.filas.items():
            if k not in ("est", "R9"):
                r[f"lam_{k}"] = float(lam[filas].sum())
        r["lam_est"] = lam[self.filas["est"]].tolist()
        r["lam_R9"] = lam[self.filas["R9"]].tolist()
        return r
