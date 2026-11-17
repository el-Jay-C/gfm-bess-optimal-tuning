"""Experimentos computacionales. Cada experimento escribe sus propios CSV en resultados/.

Uso (desde Codigos_final/):  python python/ejecutar_experimentos.py <experimento> [opciones]
  conv --metodo F1|F2|F3|F4|F4b|F5 [--n 20]   convergencia desde los mismos puntos iniciales (S1, modo R)
  modos        restricciones (R) frente a limitador en la dinámica (L)
  ablacion     efecto de agregar cada grupo de restricciones sobre las del artículo base
  pareto       frontera de calidad de tensión y sensibilidad a pérdidas
  barridos     energía reservada (continuación + multiplicadores) y tamaño de la contingencia
  robusto      una sintonía para S1-S3: min-max, valor esperado, CVaR y regla afín
  malla        refinamiento de la malla de la colocación
  optimos      reúne las sintonías finales en optimos.csv (entrada de julia/validar.jl)
  cert         amortiguamiento por autovalores y certificado de Lyapunov común (SDP)
"""
import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gfm import datos, form_directa, form_simulacion as fs, modelo  # noqa: E402
from gfm.historial import Historial  # noqa: E402
from gfm.oraculo import OraculoJulia  # noqa: E402

TH = datos.NOMBRES_THETA
TOL_FACT = 1e-3                          # violación relativa máxima para considerar factible una sintonía
p = c = esc = R = ab = args = None
WORKERS = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def multiarranque(prob, ths, nombre=""):
    """SQP de caja negra desde varios puntos; devuelve todas las corridas y la mejor factible."""
    def correr_uno(item):
        k, th0 = item
        t0, n0 = time.perf_counter(), prob.orac.n_eval
        z = fs.f2_sqp(prob, th0)
        return {"corrida": k, "z": z, "J": prob.J_real(z), "viol": prob.viol(z),
                "t_s": time.perf_counter() - t0, "n_eval": prob.orac.n_eval - n0}

    nw = getattr(prob.orac, "n_workers", 1)
    if nw > 1 and len(ths) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ths), nw)) as ex:
            filas = list(ex.map(correr_uno, enumerate(ths)))
    else:
        filas = [correr_uno(it) for it in enumerate(ths)]
    fact = [f for f in filas if f["viol"] <= TOL_FACT]
    mejor = min(fact, key=lambda f: f["J"]) if fact else min(filas, key=lambda f: (f["viol"], f["J"]))
    log(f"{nombre}: mejor J = {mejor['J']:.5f} ({len(fact)}/{len(filas)} factibles, "
        f"{sum(abs(f['J'] - mejor['J']) <= 1e-3 * mejor['J'] for f in fact)} en el mejor)")
    return filas, mejor


def activas(prob, z, s=0):
    """Nombres de las restricciones activas (|g| < 2e-3) de la sintonía z en el escenario s."""
    o = prob.orac
    m, g = o.evaluar(prob.theta(z, s), prob.idx[s], prob.lim)
    g = g.copy()
    if prob.reserva is not None:
        for nombre in ("E1_serv", "E2_serv"):
            k = o.ig[nombre]
            g[k] = (g[k] + 1) * p["bess"]["reserva"] / prob.reserva - 1
    act = [o.nombres_g[k] for k in prob.cols if g[k] + prob.margen > -2e-3]
    if prob.eps is not None and 1 - m["vdev"] / prob.eps[s] < 2e-3:
        act.append("V")
    if prob.zeta_min is not None and prob.zeta(z, s) / prob.zeta_min - 1 < 2e-3:
        act.append("zeta")
    th = prob.theta(z, s)
    G, h = datos.restricciones_estaticas(p, c, [prob.escs[s]] if prob.xi is not None else prob.escs, prob.reserva)
    filas_est = [i for nombre in prob.estaticas if nombre in fs.FILAS_EST for i in fs.FILAS_EST[nombre]]
    holg = (h - G @ th) / np.maximum(np.abs(h), 1.0)
    act += [datos.NOMBRES_EST[i] for i in filas_est if holg[i] < 2e-3]
    gt = datos.g_tension(th, ab, p, c)
    act += [n for n, v in zip(["R9_lento1", "R9_lento2", "R9_rapido1", "R9_rapido2"], gt)
            if "R9" in prob.estaticas and v > -2e-3]
    lb, ub = datos.limites(p)
    act += [f"{n}_min" for n, v, l in zip(TH, th, lb) if v - l < 1e-3 * (1 + abs(l))]
    act += [f"{n}_max" for n, v, u in zip(TH, th, ub) if u - v < 1e-3 * (1 + abs(u))]
    return " ".join(act)


def fila_sintonia(prob, z, s=0, **extra):
    m, _ = prob.orac.evaluar(prob.theta(z, s), prob.idx[s], prob.lim)
    if prob.eps is not None:
        extra["eps"] = prob.eps[s]
    else:
        extra.setdefault("eps", np.inf)
    if prob.zeta_min is not None:
        extra["zeta_S1"] = prob.zeta(z, 0)
    return {**extra, **dict(zip(TH, prob.theta(z, s))), "J": prob.J_real(z), "viol": prob.viol(z),
            "dfmax_Hz": m["dfmax_Hz"], "fmin_Hz": m["fmin_Hz"], "vdev": m["vdev"], "perdidas": m["perdidas"],
            "pred1_max": m["pred1_max"], "pred2_max": m["pred2_max"], "alivio_carga": m["alivio_carga"],
            "pb1_pico": m["pb1_pico"], "pb2_pico": m["pb2_pico"], "I1_fin": m["I1_fin"], "I2_fin": m["I2_fin"],
            "activas": activas(prob, z, s)}


# ============================================================ convergencia
def exp_conv():
    """Todas las formulaciones resuelven el mismo problema (S1, modo R) desde los mismos puntos iniciales.
    Cada corrida se guarda al terminar; si el archivo ya existe, se reanuda desde la corrida siguiente."""
    met, e0 = args.metodo, esc[0]["nombre"]
    ths = datos.puntos_iniciales(p, 20)[:args.n]
    f_conv, f_corr = R / f"conv_{met}.csv", R / f"corridas_{met}.csv"
    conv = [pd.read_csv(f_conv)] if f_corr.exists() else []
    corr = [pd.read_csv(f_corr)] if f_corr.exists() else []
    hechas = set(corr[0].corrida) if corr else set()
    with OraculoJulia(1) as orac:
        ref = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, modo="R")      # criterio común de evaluación

        def real(th):
            z = ref.z_de(np.asarray(th, float))
            return ref.J_real(z), ref.viol(z)

        def cerrar(h, k, th, t_total, **extra):
            """J y violación de cada iterado con el criterio común (modelo de Julia, todas las restricciones)."""
            d = pd.DataFrame(h.filas)
            d["J_real"], d["viol_real"] = zip(*[real(r) for r in d[TH].to_numpy(float)])
            Jr, vr = real(th)
            conv.append(d)
            corr.append(pd.DataFrame([{"formulacion": met, "corrida": k, "t_total_s": t_total,
                                       "iteraciones": len(h.filas) - 1, **dict(zip(TH, th)), **extra,
                                       "J_real": Jr, "viol_real": vr}]))
            pd.concat(conv).to_csv(f_conv, index=False)
            pd.concat(corr).to_csv(f_corr, index=False)
            log(f"{met} corrida {k}: {t_total:.1f} s, J = {Jr:.5f}, violación {vr:.1e}")

        if met in ("F1", "F2", "F3"):
            for k, th0 in enumerate(ths):
                if k in hechas:
                    continue
                orac.cache.clear()                              # sin memoria entre corridas: tiempos comparables
                orac.n_eval = 0
                h = Historial(met, k, e0)
                if met == "F1":
                    th = fs.f1_paper(orac, p, c, 0, h, semilla=k, th0=th0)
                else:
                    th = ref.theta(fs.f2_sqp(ref, th0, h) if met == "F2" else fs.f3_pso(ref, h, semilla=k, th0=th0))
                cerrar(h, k, th, h.t, n_eval=orac.n_eval, t_solver_s=h.t)
        else:
            cfg = {"F4": dict(metodo="trapecio"), "F4b": dict(metodo="trapecio", hess="lbfgs", max_iter=500),
                   "F5": dict(metodo="disparo", max_iter=250)}[met]
            P = form_directa.ProblemaDirecto(p, c, [esc[0]], modo="R", **cfg)
            log(f"{met}: NLP construido en {P.t_construccion:.1f} s ({P.nw} variables, {P.ng} restricciones)")
            for k, th0 in enumerate(ths):
                if k in hechas:
                    continue
                h = Historial(met, k, e0)
                r = P.resolver(th0, h)
                cerrar(h, k, r["th"], r["t_inicial_s"] + r["t_s"], t_solver_s=r["t_s"], t_inicial_s=r["t_inicial_s"],
                       t_construccion_s=r["t_construccion_s"], n_var=r["n_var"], n_res=r["n_res"],
                       estado=r["estado"], exito=r["exito"], J_modelo=r["J"])


# ============================================================ restricciones frente a limitador
def exp_modos():
    ths = datos.puntos_iniciales(p, 20)[:args.n]
    filas = []
    with OraculoJulia(WORKERS) as orac:
        for modo in ("R", "L"):
            pr = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, modo=modo)
            corr, mejor = multiarranque(pr, ths, f"modo {modo}")
            Js = np.array([f["J"] for f in corr if f["viol"] <= TOL_FACT])
            filas.append(fila_sintonia(pr, mejor["z"], modo=modo, metodo="F2", n_factibles=len(Js),
                                       n_mejor=int(np.sum(Js <= mejor["J"] * (1 + 1e-3))),
                                       J_cuencas=" ".join(f"{v:.4f}" for v in sorted(set(np.round(Js, 4))))))
            P = form_directa.ProblemaDirecto(p, c, [esc[0]], "trapecio", modo=modo)
            r = P.resolver(pr.theta(mejor["z"]))
            z = pr.z_de(r["th"])
            filas.append(fila_sintonia(pr, z, modo=modo, metodo="F4", estado=r["estado"], t_s=r["t_s"],
                                       J_modelo=r["J"], **{k: v for k, v in r.items() if k.startswith("lam_")
                                                           and not isinstance(v, list)},
                                       lam_est=" ".join(f"{v:.3e}" for v in r["lam_est"]),
                                       lam_R9=" ".join(f"{v:.3e}" for v in r["lam_R9"])))
            log(f"modo {modo} F4: {r['estado']} J = {r['J']:.5f} (Julia {filas[-1]['J']:.5f})")
    pd.DataFrame(filas).to_csv(R / "modos.csv", index=False)


# ============================================================ ablación de restricciones
def exp_ablacion():
    ths = datos.puntos_iniciales(p, 20)[:args.n]
    casos = [("Artículo: cotas + P ≤ Pmax", dict(grupos=(), estaticas=(), p_max=True)),
             ("+ R2 frecuencia y R6 tensión", dict(grupos=("R2", "R6"), estaticas=(), p_max=True)),
             ("+ R3 corriente", dict(grupos=("R2", "R6", "R3"), estaticas=(), p_max=True)),
             ("+ R4 límites del BESS", dict(grupos=("R2", "R6", "R3", "R4"), estaticas=(), p_max=True)),
             ("+ R5 energía de servicio", dict(grupos=("R2", "R6", "R3", "R4", "R5"), estaticas=(), p_max=True)),
             ("+ R7-R8 inercia", dict(grupos=("R2", "R6", "R3", "R4", "R5"), estaticas=("R7", "R8"), p_max=True)),
             ("+ R9 control de tensión", dict(estaticas=("R7", "R8", "R9"), p_max=True)),
             ("+ R11 validez fasorial", dict(p_max=True)),
             ("(P): + ε de tensión, sin P ≤ Pmax", dict(eps_V=None))]
    filas = []
    with OraculoJulia(WORKERS) as orac:
        completo = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, modo="R")
        _, sol_P = multiarranque(completo, ths, "(P) completo")
        th_P, previa = completo.theta(sol_P["z"]), []
        for nombre, kw in casos:
            pr = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, modo="R", **{"eps_V": np.inf, **kw})
            _, mejor = multiarranque(pr, ths + previa + [th_P], nombre)
            previa = [pr.theta(mejor["z"])]
            f = fila_sintonia(pr, mejor["z"], caso=nombre)
            f["viol_P"] = completo.viol(completo.z_de(pr.theta(mejor["z"])))     # violación del problema completo
            filas.append(f)
            pd.DataFrame(filas).to_csv(R / "ablacion.csv", index=False)


# ============================================================ frontera de Pareto
def exp_pareto():
    ths = datos.puntos_iniciales(p, 20)[:args.n]
    o, filas, previa = p["optimizacion"], [], []
    P = form_directa.ProblemaDirecto(p, c, [esc[0]], modo="R")
    with OraculoJulia(WORKERS) as orac:
        for eps_V, wl in [(e, o["wl"]) for e in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, np.inf)] + [(1.0, 0.0), (1.0, 1.0)]:
            pr = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, eps_V=eps_V, pesos_J=(o["wf"], 0, wl))
            _, mejor = multiarranque(pr, ths + previa, f"ε={eps_V} wl={wl}")
            # Sólo el objetivo cambia para los dos casos de wl; el barrido ε reutiliza un único NLP.
            directo = P if wl == o["wl"] else form_directa.ProblemaDirecto(
                p, c, [esc[0]], pesos_J=(o["wf"], 0, wl))
            base = directo.resolver(pr.theta(mejor["z"]), eps_V=eps_V)
            previa = [base["th"]]
            f = fila_sintonia(pr, pr.z_de(base["th"]), eps_V=eps_V, wl=wl, lam_V=base["lam_V"],
                              J_modelo=base["J"], estado=base["estado"], eps=base["eps"][0])
            filas.append(f)
            pd.DataFrame(filas).to_csv(R / "pareto.csv", index=False)
            if eps_V == 1.0 and wl == o["wl"]:
                laterales = []
                for e in (0.98, 1.02):
                    res = P.resolver(base["th"], eps_V=e)
                    ref = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, eps_V=e)
                    lat = fila_sintonia(ref, ref.z_de(res["th"]), eps_V=e, wl=wl, lam_V=res["lam_V"],
                                        J_modelo=res["J"], estado=res["estado"])
                    filas.append(lat)
                    laterales.append(res)
                    pd.DataFrame(filas).to_csv(R / "pareto.csv", index=False)
                dif = ((laterales[1]["J"] - laterales[0]["J"]) / 0.04
                       if all(res["exito"] for res in laterales) and base["exito"] else np.nan)
                pred = -base["lam_V"] * P.vdev0[0]
                f.update(dJ_deps_multiplicador=pred, dJ_deps_diferencias=dif,
                         error_relativo=abs(pred - dif) / max(abs(dif), 1e-12),
                         mismo_activo=len({f["activas"], filas[-2]["activas"], filas[-1]["activas"]}) == 1)
                pd.DataFrame(filas).to_csv(R / "pareto.csv", index=False)


# ============================================================ barridos
def exp_barridos():
    ths = datos.puntos_iniciales(p, 20)[:args.n]
    r0 = p["bess"]["reserva"]
    filas = []
    P = form_directa.ProblemaDirecto(p, c, [esc[0]], "trapecio", modo="R")
    with OraculoJulia(WORKERS) as orac:
        # Energía reservada: multiarranque con SQP (más la solución del punto anterior, continuación) y luego
        # colocación desde esa solución, que entrega los multiplicadores. El multiplicador se contrasta con la
        # pendiente por diferencias centradas (±1 %), resueltas con la colocación desde la misma solución.
        previa = []
        for rho in (0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.10):
            pr = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, modo="R", reserva=rho)
            _, mejor = multiarranque(pr, ths + previa, f"reserva {rho}")
            base = P.resolver(pr.theta(mejor["z"]), reserva=rho)
            previa = [base["th"]]
            Jc = [P.resolver(base["th"], reserva=rho * fr) for fr in (0.99, 1.01)]
            ok = base["exito"] and all(r["exito"] for r in Jc)
            act = [activas(pr, pr.z_de(base["th"]))]
            for fr, res in zip((0.99, 1.01), Jc):
                ref = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, reserva=rho * fr)
                act.append(activas(ref, ref.z_de(res["th"])))
            dif = (Jc[1]["J"] - Jc[0]["J"]) / (0.02 * rho) if ok else np.nan
            # dJ*/dρ = -Σ_i λ_i dE_res,i/dρ, con λ de las filas de energía de servicio y de R8a
            dE = c["Eres"] / r0
            pred = -(base["lam_E0"] * dE[0] + base["lam_E1"] * dE[1]
                     + base["lam_est"][1] * dE[0] + base["lam_est"][2] * dE[1])
            filas.append(fila_sintonia(pr, pr.z_de(base["th"]), barrido="reserva", valor=rho, estado=base["estado"],
                                       J_modelo=base["J"], lam_E1=base["lam_E0"], lam_E2=base["lam_E1"],
                                       dJ_drho_multiplicador=pred,
                                       dJ_drho_diferencias=dif, mismo_activo=len(set(act)) == 1,
                                       error_relativo=abs(pred - dif) / max(abs(dif), 1e-12)))
            log(f"reserva {rho}: J = {base['J']:.5f}, dJ/dρ multiplicador {pred:.4f}, diferencias "
                f"{filas[-1]['dJ_drho_diferencias']:.4f}")
            pd.DataFrame(filas).to_csv(R / "barridos.csv", index=False)
        # Tamaño de la contingencia (el requisito de inercia R7 sigue fijado por los 50 MW de diseño)
        for modo in ("R", "L"):
            previa = []
            for dP in (20.0, 30.0, 40.0, 50.0, 60.0):
                k = orac.nuevo(5, dP)
                e = {**esc[0], "dP": dP / c["Sb"]}
                pr = fs.ProblemaCN(orac, p, c, [e], [k], ab, modo=modo)
                _, mejor = multiarranque(pr, ths + previa, f"dP {dP} modo {modo}")
                previa = [pr.theta(mejor["z"])]
                filas.append(fila_sintonia(pr, mejor["z"], barrido=f"dP_{modo}", valor=dP))
                pd.DataFrame(filas).to_csv(R / "barridos.csv", index=False)


# ============================================================ una sintonía para varios escenarios
def exp_robusto():
    ths = datos.puntos_iniciales(p, 20)[:args.n]
    pi = [e["prob"] for e in p["escenario"]]
    xi = [(p["gfm"]["P_fv"][1] - e.get("P_fv2", p["gfm"]["P_fv"][1])) / 10.0 for e in p["escenario"]]
    filas, sem = [], {}
    with OraculoJulia(WORKERS) as orac:
        for k, e in enumerate(esc):                               # óptimo de cada escenario por separado
            pr = fs.ProblemaCN(orac, p, c, [e], [k], ab, modo="R")
            _, mejor = multiarranque(pr, ths, f"óptimo de {e['nombre']}")
            sem[k] = pr.theta(mejor["z"])
            f = fila_sintonia(pr, mejor["z"], modo="R", sintonia=f"opt_{e['nombre']}", criterio="un escenario")
            Js = []
            for j in range(3):
                ref = fs.ProblemaCN(orac, p, c, [esc[j]], [j], ab)
                z = ref.z_de(sem[k])
                Js.append(ref.J_real(z))
                f.update({f"J_S{j+1}": Js[-1], f"viol_S{j+1}": ref.viol(z)})
            f.update(J_max=max(Js), J_esp=float(np.dot(pi, Js)))
            filas.append(f)
            pd.DataFrame(filas).to_csv(R / "robusto.csv", index=False)
        casos = [("minmax", dict(agregacion="minmax")), ("esperado", dict(agregacion="esperado", pesos=pi)),
                 ("cvar80", dict(agregacion="cvar", pesos=pi, alpha=0.8)),
                 ("afin_minmax", dict(agregacion="minmax", xi=xi))]
        for modo in ("R", "L"):
            previa = []
            for nombre, kw in casos:
                if modo == "L" and "xi" in kw:
                    continue
                pr = fs.ProblemaCN(orac, p, c, esc, [0, 1, 2], ab, modo=modo,
                                   margen=0.005, zeta_min=0.05, **kw)
                semillas = ths[:2] + list(sem.values()) + previa
                if "xi" in kw:
                    semillas += [np.concatenate([sem[k], sem[2] - sem[k]]) for k in (0, 1)]
                corr, mejor = multiarranque(pr, semillas, f"{nombre} {modo}")
                z = mejor["z"]
                previa = [pr.theta(z)]
                Js = [pr.J_esc(z, s) for s in range(3)]
                f = fila_sintonia(pr, z, modo=modo, sintonia=nombre, criterio=nombre,
                                  J_S1=Js[0], J_S2=Js[1], J_S3=Js[2], J_max=max(Js), J_esp=float(np.dot(pi, Js)),
                                  viol_min_corridas=min(f["viol"] for f in corr))
                for k in range(3):
                    ref = fs.ProblemaCN(orac, p, c, [esc[k]], [k], ab, modo=modo, margen=0.005, zeta_min=0.05)
                    f[f"viol_S{k+1}"] = ref.viol(ref.z_de(pr.theta(z, k)))
                if "xi" in kw:
                    d = np.asarray(z[8:16]) * (pr.ub - pr.lb)
                    f.update({f"d{n}": v for n, v in zip(TH, d)})
                    f["activas"] = " | ".join(activas(pr, z, s) for s in range(3))
                filas.append(f)
                pd.DataFrame(filas).to_csv(R / "robusto.csv", index=False)
                if args.rapido:
                    return


# ============================================================ refinamiento de malla
def exp_malla():
    mod = pd.read_csv(R / "modos.csv")
    th0 = mod[(mod.modo == "R") & (mod.metodo == "F4")].iloc[0][TH].to_numpy(float)
    f_malla = R / "malla.csv"
    prev = (pd.read_csv(f_malla)
            if f_malla.exists() and f_malla.stat().st_mtime >= (R / "modos.csv").stat().st_mtime
            else pd.DataFrame())
    filas = prev.to_dict("records") if not prev.empty else []
    hechos = {int(f) for f in prev["factor"]} if not prev.empty else set()
    with OraculoJulia(1) as orac:
        ref = fs.ProblemaCN(orac, p, c, [esc[0]], [0], ab, modo="R")
        for factor in (1, 2, 4):
            if factor in hechos:
                continue
            P = form_directa.ProblemaDirecto(p, c, [esc[0]], "trapecio", modo="R", factor=factor)
            r = P.resolver(th0)
            z = ref.z_de(r["th"])
            filas.append({"factor": factor, "nodos": len(P.t), "n_var": r["n_var"], "estado": r["estado"],
                          "t_s": r["t_s"], "t_construccion_s": r["t_construccion_s"], "J_modelo": r["J"],
                          "J_real": ref.J_real(z), "viol_real": ref.viol(z), **dict(zip(TH, r["th"]))})
            log(f"malla ×{factor}: {r['estado']} J = {r['J']:.6f}, Julia {filas[-1]['J_real']:.6f}, "
                f"violación {filas[-1]['viol_real']:.1e}, {r['t_s']:.0f} s")
            pd.DataFrame(filas).to_csv(f_malla, index=False)


# ============================================================ sintonías finales
def exp_optimos():
    filas = [{"sintonia": "inicial", "modo": "R", "escenario": esc[0]["nombre"],
              **dict(zip(TH, p["optimizacion"]["theta_inicial"]))}]
    f1 = pd.read_csv(R / "corridas_F1.csv").sort_values("J_real").iloc[0]
    filas.append({"sintonia": "F1_articulo", "modo": "R", "escenario": esc[0]["nombre"], **{n: f1[n] for n in TH}})
    mod = pd.read_csv(R / "modos.csv")
    for modo, nombre in (("R", "opt_R"), ("L", "opt_L")):
        f = mod[(mod.modo == modo) & mod.metodo.isin(["F2", "F4"]) &
                (mod.viol <= TOL_FACT)].sort_values("J").iloc[0]
        filas.append({"sintonia": nombre, "modo": modo, "escenario": esc[0]["nombre"], **{n: f[n] for n in TH}})
    rob = pd.read_csv(R / "robusto.csv")
    for s, modo in [(s, "R") for s in ("minmax", "esperado", "cvar80", "afin_minmax")] + [("minmax_L", "L")]:
        f = rob[(rob.sintonia == ("minmax" if s == "minmax_L" else s)) & (rob.modo == modo)].iloc[0]
        filas.append({"sintonia": s, "modo": modo, "escenario": esc[0]["nombre"], **{n: f[n] for n in TH},
                      **({f"d{n}": f[f"d{n}"] for n in TH} if s == "afin_minmax" else {})})
    df = pd.DataFrame(filas)
    for n in TH:
        if f"d{n}" not in df:
            df[f"d{n}"] = 0.0
    df.fillna(0.0).to_csv(R / "optimos.csv", index=False)
    log(f"optimos.csv con {len(df)} sintonías")


# ============================================================ certificados
def _cert_uno(tarea):
    from gfm import certificado
    c_loc, esc_loc, th_names, r = tarea
    As, fila = [], {"sintonia": r["sintonia"], "modo": r["modo"]}
    th_base = np.array([r[n] for n in th_names], float)
    dth = np.array([r[f"d{n}"] for n in th_names], float)
    for k, e in enumerate(esc_loc):
        th = th_base + (0.0 if k < 2 else 1.0) * dth
        A, resid = certificado.linealizar(c_loc, {**e, "lim": float(r["modo"] == "L")}, th)
        ev = np.linalg.eigvals(A)
        osc = ev[np.abs(ev.imag) > 1e-6]
        fila[f"zeta_{e['nombre']}"] = float((-osc.real / np.abs(osc)).min()) if len(osc) else 1.0
        fila[f"alpha_{e['nombre']}"] = float(-ev.real.max())
        fila[f"residuo_{e['nombre']}"] = resid
        As.append(A)
    fila["zeta_min"] = min(fila[f"zeta_{e['nombre']}"] for e in esc_loc)
    fila["alpha_eig_min"] = min(fila[f"alpha_{e['nombre']}"] for e in esc_loc)
    fila["alpha_LMI_comun"] = certificado.tasa_certificada_comun(As)
    fila["lmi_comun_existe"] = bool(np.isfinite(fila["alpha_LMI_comun"]))
    fila["zeta_LMI_S1"] = certificado.amortiguamiento_certificado(As[0])
    return fila


def exp_cert():
    opt = pd.read_csv(R / "optimos.csv")
    tareas = [(c, esc, TH, r.to_dict()) for _, r in opt.iterrows()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(tareas), os.cpu_count() or 1)) as ex:
        filas = list(ex.map(_cert_uno, tareas))
    for fila in filas:
        log(f"{fila['sintonia']}: ζ mín = {fila['zeta_min']:.3f}, α común (SDP) = {fila['alpha_LMI_comun']:.4f}, "
            f"α mín por autovalores = {fila['alpha_eig_min']:.4f}")
    pd.DataFrame(filas).to_csv(R / "certificados.csv", index=False)


def main():
    global p, c, esc, R, ab, args, WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--metodo", default="F2")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rapido", action="store_true", help="en robusto termina tras el primer caso multiescenario R")
    ap.add_argument("--res", default=None, help="carpeta de resultados (por defecto resultados/)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("GFM_WORKERS", "0")),
                    help="procesos de Julia en paralelo (0 = mitad de hilos de CPU)")
    args = ap.parse_args()
    WORKERS = max(1, (os.cpu_count() or 2) // 2) if args.workers <= 0 else args.workers
    p, c, esc = datos.cargar()
    R = datos.RESULTADOS if args.res is None else Path(args.res)
    R.mkdir(exist_ok=True, parents=True)
    ab = modelo.sensibilidad_tension(c, esc[0])
    {"conv": exp_conv, "modos": exp_modos, "ablacion": exp_ablacion, "pareto": exp_pareto, "barridos": exp_barridos,
     "robusto": exp_robusto, "malla": exp_malla, "optimos": exp_optimos, "cert": exp_cert}[args.exp]()
    log(f"Listo: {args.exp}")


if __name__ == "__main__":
    main()
