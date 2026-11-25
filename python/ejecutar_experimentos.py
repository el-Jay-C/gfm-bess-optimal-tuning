"""Experimentos computacionales. Cada experimento escribe sus propios CSV en resultados/.

Uso (desde Codigos_final/):  python python/ejecutar_experimentos.py <experimento> [opciones]
  conv --metodo F1|F2|F3|F4|F4b|F5 [--n 20]   convergencia en S1, modo L
  modos        restricciones (R) frente a limitador en la dinámica (L)
  ablacion     efecto de agregar cada grupo de restricciones sobre las del artículo base
  pareto       sensibilidad a pérdidas y al multiplicador de la cota de tensión
  barridos     energía reservada (continuación + multiplicadores) y tamaño de la contingencia
  robusto      sintonías para S1-S4: min-max, valor esperado, CVaR y dos juegos
  rango --problema L_todos|L_S1|R_S3   anclajes y frontera de tensión
  kappa        selección difusa de la cota de tensión
  fase1        mínima violación en problemas de factibilidad
  sens --caso C  sensibilidad a parámetros del modelo
  malla        refinamiento de la malla de la colocación
  optimos      reúne las sintonías finales en optimos.csv (entrada de julia/validar.jl)
  cert         amortiguamiento por autovalores y certificado de Lyapunov común (SDP)
"""
import argparse
import concurrent.futures
import os
import re
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from gfm import datos, form_directa, form_simulacion as fs, modelo  # noqa: E402
from gfm.historial import Historial  # noqa: E402
from gfm.oraculo import OraculoJulia  # noqa: E402

TH = datos.NOMBRES_THETA
TOL_FACT = 1e-3                          # violación relativa máxima para considerar factible una sintonía
p = c = esc = R = ab = args = None
WORKERS = 1
PROBLEMAS = {"L_todos": [0, 1, 2, 3], "L_S1": [0], "R_S3": [2]}
KAPPAS = (0, 0.1, 0.25, 0.5, 0.75, 1)
SENS = {"klim1": ("gfm", "k_lim", 1.0), "klim4": ("gfm", "k_lim", 4.0),
        "kp0": ("gfm", "kp_lim", 0.0), "kp1": ("gfm", "kp_lim", 1.0),
        "fracP0": ("red", "frac_P", 0.0), "fracP1": ("red", "frac_P", 1.0),
        "Kqmin5": ("optimizacion", "Kq_lim", 5.0),
        "nsep3": ("gfm", "n_sep", 3.0), "nsep8": ("gfm", "n_sep", 8.0)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cota(problema, ks):
    tabla = pd.read_csv(R / "rango_elegido.csv")
    filas = tabla[tabla.problema == problema].set_index("escenario")
    return None if filas.empty else filas.loc[[esc[k]["nombre"] for k in ks], "cota"].to_numpy(float)


def semillas():
    return datos.puntos_iniciales(p, max(20, args.n))[:args.n]


def semilla_conjunta():
    ruta = R / "rango_L_todos.csv"
    elegida = R / "rango_elegido.csv"
    if ruta.exists() and elegida.exists():
        d = pd.read_csv(ruta)
        d = d[(d.tipo == "rango") & (d.viol <= TOL_FACT)]
        if not d.empty:
            kappa = pd.read_csv(elegida).kappa.iloc[0]
            factibles = d[d.kappa <= kappa + 1e-9]
            mejor = factibles if not factibles.empty else d
            return [mejor.loc[(mejor.kappa - kappa).abs().idxmin(), TH].to_numpy(float)]
    return []


def multiarranque(prob, ths, nombre="", **opciones):
    """SQP de caja negra desde varios puntos; devuelve todas las corridas y la mejor factible."""
    def correr_uno(item):
        k, th0 = item
        t0, n0 = time.perf_counter(), prob.orac.n_eval
        z = fs.f2_sqp(prob, th0, **opciones)
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
    p, c, ab = prob.p, prob.c, prob.ab
    m, g = o.evaluar(prob.theta(z), prob.idx[s], prob.lim)
    g = g.copy()
    if prob.reserva is not None:
        for nombre in ("E1_serv", "E2_serv"):
            k = o.ig[nombre]
            g[k] = (g[k] + 1) * p["bess"]["reserva"] / prob.reserva - 1
    act = [o.nombres_g[k] for k in prob.cols if g[k] + prob.margen > -2e-3]
    if prob.cota_V is not None and (prob.cota_V[s] - m["vdev"]) < 2e-3 * max(prob.cota_V[s], 1e-12):
        act.append("V")
    if prob.con_zeta[s] and prob.zeta(z, s) / prob.zeta_min - 1 < 2e-3:
        act.append("zeta")
    th = prob.theta(z)
    G, h = datos.restricciones_estaticas(p, c, prob.escs, prob.reserva)
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
    m, _ = prob.orac.evaluar(prob.theta(z), prob.idx[s], prob.lim)
    extra.update(cota_V=prob.cota_V[s] if prob.cota_V is not None else np.nan, zeta=prob.zeta(z, s),
                 zeta_min=prob.zeta_min, V_min=m["V_min"], V_max=m["V_max"])
    return {**extra, **dict(zip(TH, prob.theta(z))), "J": prob.J_real(z), "viol": prob.viol(z),
            "dfmax_Hz": m["dfmax_Hz"], "fmin_Hz": m["fmin_Hz"], "vdev": m["vdev"], "perdidas": m["perdidas"],
            "pred1_max": m["pred1_max"], "pred2_max": m["pred2_max"], "alivio_carga": m["alivio_carga"],
            "pb1_pico": m["pb1_pico"], "pb2_pico": m["pb2_pico"], "I1_fin": m["I1_fin"], "I2_fin": m["I2_fin"],
            "activas": " | ".join(activas(prob, z, k) for k in range(len(prob.idx)))}


def problema(orac, ks, modo="L", **kw):
    return fs.ProblemaCN(orac, p, c, [esc[k] for k in ks], ks, ab, modo=modo, **kw)


def fase_uno(prob, ths):
    """Minimiza la holgura común de las restricciones normalizadas."""
    def g(x):
        return prob.g(x[:-1]) + x[-1]

    filas = []
    for th in ths:
        z = np.clip(prob.z_de(th), 0, 1)
        x0 = np.r_[z, prob.viol(z)]
        filas.append(dict(z=z, viol=x0[-1]))
        if x0[-1] <= 1e-7:
            break
        r = minimize(lambda x: x[-1], x0, method="SLSQP", bounds=prob.cotas + [(0, None)],
                     constraints=[{"type": "ineq", "fun": g}], jac=lambda x: np.r_[np.zeros(prob.dim), 1.0],
                     options={"maxiter": 60, "ftol": 1e-6, "eps": 1e-5})
        z, viol = r.x[:-1], prob.viol(r.x[:-1])
        filas.append(dict(z=z, viol=viol))
        if viol <= 1e-7:
            break
    mejor = min(filas, key=lambda f: f["viol"])
    z, t = mejor["z"], mejor["viol"]
    fijan = [n for n, v in zip(prob.nombres_g(), prob.g(z)) if v + t <= 2e-4]
    fijan += [f"{n}_{lado}" for n, v in zip(TH, z) for lado, b in (("min", 0), ("max", 1))
              if abs(v - b) <= 2e-4]
    mejor.update(t_estrella=t, fijan=" ".join(fijan))
    return mejor


def anclajes(orac, ks, modo, ths):
    kw = dict(modo=modo, margen=0.005 if len(ks) > 1 else 0, agregacion="minmax")
    libre = problema(orac, ks, **kw)
    sqp = dict(ftol=1e-6, eps=1e-5)
    _, sol = multiarranque(libre, ths, "anclaje libre", **sqp)
    if sol["viol"] > TOL_FACT:
        fase = problema(orac, ks, modo, margen=kw["margen"])
        recuperada = fase_uno(fase, ths + [libre.theta(sol["z"])])
        z = libre.z_de(fase.theta(recuperada["z"]))
        if libre.viol(z) < sol["viol"]:
            sol = dict(z=z, viol=libre.viol(z), J=libre.J_real(z))
        if sol["viol"] > TOL_FACT:
            return libre, sol["z"], None, None, None
        _, reintento = multiarranque(libre, [libre.theta(sol["z"])], "anclaje libre desde fase I", **sqp)
        if reintento["viol"] <= TOL_FACT and reintento["J"] < sol["J"]:
            sol = reintento
    zlibre = sol["z"]
    thlibre = libre.theta(zlibre)

    def vdev(th):
        return np.array([orac.evaluar(th, k, libre.lim)[0]["vdev"] for k in ks])

    Vlibre = vdev(thlibre)
    tension = problema(orac, ks, V_libre=np.maximum(Vlibre, 1e-12), **kw)
    corr, _ = multiarranque(tension, ths + [thlibre], "anclaje de tensión", **sqp)
    fact = [f for f in corr if f["viol"] <= TOL_FACT and np.all(vdev(tension.theta(f["z"])) <= Vlibre)]
    if not fact:
        log("Se conserva el anclaje libre como solución factible del anclaje de tensión")
    thestrella = tension.theta(min(fact, key=lambda f: f["J"])["z"]) if fact else thlibre
    return libre, zlibre, libre.z_de(thestrella), vdev(thestrella), Vlibre


def fila_rango(pr, z, Vestrella, Vlibre, **extra):
    fila = {**extra, **dict(zip(TH, pr.theta(z))), "J": pr.J_real(z), "viol": pr.viol(z)}
    for s, k in enumerate(pr.idx):
        m = pr.orac.evaluar(pr.theta(z), k, pr.lim)[0]
        suf = f"S{k+1}"
        fila.update({f"J_{suf}": pr.J_esc(z, s), f"vdev_{suf}": m["vdev"],
                     f"V_estrella_{suf}": Vestrella[s], f"V_libre_{suf}": Vlibre[s],
                     f"zeta_{suf}": pr.zeta(z, s), f"V_min_{suf}": m["V_min"], f"V_max_{suf}": m["V_max"],
                     f"activas_{suf}": activas(pr, z, s)})
    return fila


def punto_rango(libre, zl, ze, Ve, Vl, kappa, anterior=None):
    ks = libre.idx
    cota_V = Ve + kappa * (Vl - Ve)
    pr = problema(libre.orac, ks, "L" if libre.lim else "R", margen=libre.margen,
                  agregacion="minmax", cota_V=cota_V)
    ths = [libre.theta(z) for z in (ze, zl, anterior) if z is not None]
    ths = list({tuple(th): th for th in ths}.values())
    _, mejor = multiarranque(pr, ths, f"κ = {kappa:g}", ftol=1e-6, eps=1e-5)
    for th in ths:
        z = pr.z_de(th)
        J, viol = pr.J_real(z), pr.viol(z)
        if viol <= TOL_FACT and (mejor["viol"] > TOL_FACT or J < mejor["J"]):
            mejor = dict(z=z, J=J, viol=viol)
    return pr, mejor["z"]


def exp_rango():
    nombre = args.problema
    ks = PROBLEMAS[nombre]
    ruta = R / f"rango_{nombre}.csv"
    with OraculoJulia(WORKERS) as orac:
        libre, zl, ze, Ve, Vl = anclajes(orac, ks, nombre[0], semillas())
        if ze is None:
            log(f"{nombre}: no se encontró sintonía factible, violación {libre.viol(zl):.3g}")
            vacio = np.full(len(ks), np.nan)
            filas = [fila_rango(libre, zl, vacio, vacio, tipo="infactible", kappa=np.nan)]
        else:
            filas = [fila_rango(libre, zl, Ve, Vl, tipo="libre", kappa=np.nan),
                     fila_rango(libre, ze, Ve, Vl, tipo="tension", kappa=np.nan)]
            pd.DataFrame(filas).to_csv(ruta, index=False)
            anterior = None
            for kappa in KAPPAS:
                pr, anterior = punto_rango(libre, zl, ze, Ve, Vl, kappa, anterior)
                filas.append(fila_rango(pr, anterior, Ve, Vl, tipo="rango", kappa=kappa))
                pd.DataFrame(filas).to_csv(ruta, index=False)
        pd.DataFrame(filas).to_csv(ruta, index=False)


def exp_kappa():
    rangos = {n: pd.read_csv(R / f"rango_{n}.csv") for n in PROBLEMAS}
    for nombre, tabla in rangos.items():
        if tabla.tipo.tolist() == ["infactible"]:
            continue
        if set(tabla.loc[tabla.tipo == "rango", "kappa"]) != set(KAPPAS):
            raise SystemExit(f"El barrido {nombre} está incompleto")
    d = rangos["L_todos"]
    d = d[(d.tipo == "rango") & (d.viol <= TOL_FACT) & np.isfinite(d.J)].sort_values("kappa")
    if d.empty or d.kappa.iloc[0] != 0 or d.kappa.iloc[-1] != 1:
        raise SystemExit("κ requiere extremos factibles para L_todos")
    kappa = 0.0
    J0, J1 = d.J.iloc[[0, -1]]
    if not np.isclose(J0, J1):
        malla = np.arange(101) / 100
        mu_J = (J0 - np.interp(malla, d.kappa, d.J)) / (J0 - J1)
        kappa = malla[np.argmax(np.minimum(np.clip(mu_J, 0, 1), 1 - malla))]
    filas = []
    for nombre, ks in PROBLEMAS.items():
        ancla = rangos[nombre].loc[lambda d: (d.tipo == "tension") & (d.viol <= TOL_FACT)]
        if ancla.empty:
            continue
        for k in ks:
            Ve, Vl = ancla.iloc[0][[f"V_estrella_S{k+1}", f"V_libre_S{k+1}"]]
            filas.append(dict(problema=nombre, escenario=esc[k]["nombre"], V_estrella=Ve, V_libre=Vl,
                              kappa=kappa, cota=Ve + kappa * (Vl - Ve)))
    pd.DataFrame(filas).to_csv(R / "rango_elegido.csv", index=False)
    log(f"κ elegido = {kappa:g}")


def exp_fase1():
    filas = []
    with OraculoJulia(WORKERS) as orac:
        casos = [(f"R_S{k+1}", "R", [k], [esc[k]]) for k in range(len(esc))]
        casos.append(("R_S1_S3", "R", [0, 1, 2], esc[:3]))
        for s, dP in ((5, 40), (6, 50)):
            k = orac.nuevo(5, dP, (0.2, 0.2), 35)
            e = dict(nombre=f"S{s}", dP=dP / c["Sb"], pb0=esc[2]["pb0"])
            casos.append((f"L_S1_S4_S{s}", "L", [*range(len(esc)), k], [*esc, e]))
        for nombre, modo, ks, es in casos:
            pr = fs.ProblemaCN(orac, p, c, es, ks, ab, modo=modo, margen=0.005 if len(ks) > 1 else 0)
            ths = (semilla_conjunta() if modo == "L" else []) + semillas()
            r = fase_uno(pr, ths)
            filas.append(fila_sintonia(pr, r["z"], problema=nombre, t_estrella=r["t_estrella"], fijan=r["fijan"]))
            log(f"{nombre}: t* = {r['t_estrella']:.5g}, {r['fijan']}")
            pd.DataFrame(filas).to_csv(R / "fase1.csv", index=False)


def preparar_sens(caso, carpeta):
    grupo, clave, valor = SENS[caso]
    origen = Path(os.environ.get("GFM_DATOS", datos.RAIZ / "datos")) / "parametros.toml"
    texto = origen.read_text()
    if clave == "Kq_lim":
        valor = [valor, tomllib.loads(texto)[grupo][clave][1]]
    patron = rf"(?ms)(^\[{grupo}\][^\n]*\n(?:(?!^\[).)*?^{clave}\s*=\s*)[^#\n]+"
    texto, n = re.subn(patron, rf"\g<1>{valor} ", texto, count=1)
    if n != 1:
        raise ValueError(f"No se encontró {grupo}.{clave} en {origen}")
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "parametros.toml").write_text(texto)
    os.environ["GFM_DATOS"] = str(carpeta.resolve())


def exp_sens():
    ks = PROBLEMAS["L_todos"]
    kappa = float(pd.read_csv(R / "rango_elegido.csv").kappa.iloc[0])
    with OraculoJulia(WORKERS) as orac:
        libre, zl, ze, Ve, Vl = anclajes(orac, ks, "L", semillas() + semilla_conjunta())
        if ze is None:
            pr, z = libre, zl
            Ve = Vl = np.full(len(ks), np.nan)
        else:
            pr, z = punto_rango(libre, zl, ze, Ve, Vl, kappa)
        fila = fila_rango(pr, z, Ve, Vl, caso=args.caso, kappa=kappa)
        for s, k in enumerate(ks):
            fila[f"cota_S{k+1}"] = pr.cota_V[s] if pr.cota_V is not None else np.nan
        if (R / "robusto.csv").exists():
            rob = pd.read_csv(R / "robusto.csv")
            base = rob[(rob.sintonia == "minmax") & (rob.modo == "L")] if "J_S4" in rob else rob.iloc[:0]
            if not base.empty:
                th = base.iloc[0][TH].to_numpy(float)
                fila["viol_base"] = pr.viol(pr.z_de(th))
                for s, k in enumerate(ks):
                    cv = None if pr.cota_V is None else pr.cota_V[s:s+1]
                    ref = problema(orac, [k], margen=pr.margen, cota_V=cv)
                    fila[f"viol_base_S{k+1}"] = ref.viol(ref.z_de(th))
        pd.DataFrame([fila]).to_csv(R / f"sens_{args.caso}.csv", index=False)


# ============================================================ convergencia
def exp_conv():
    """Evalúa las seis formulaciones en S1, modo L, desde los mismos puntos iniciales.
    Cada corrida se guarda al terminar; si el archivo ya existe, se reanuda desde la corrida siguiente."""
    met, e0 = args.metodo, esc[0]["nombre"]
    ths = semillas()
    cv = cota("L_S1", [0])
    f_conv, f_corr = R / f"conv_{met}.csv", R / f"corridas_{met}.csv"
    conv, corr = [], []
    if f_corr.exists() and f_conv.exists():
        prev = pd.read_csv(f_corr)
        if "modo_ref" in prev and (prev.modo_ref == "L").all():
            corr = [prev]
            historial = pd.read_csv(f_conv)
            conv = [historial[historial.corrida.isin(prev.corrida)]]
    hechas = set(corr[0].corrida) if corr else set()
    with OraculoJulia(1) as orac:
        ref = problema(orac, [0], cota_V=cv)

        def real(th):
            z = ref.z_de(np.asarray(th, float))
            return ref.J_real(z), ref.viol(z)

        def cerrar(h, k, th, t_total, **extra):
            """J y violación de cada iterado con el criterio común (modelo de Julia, todas las restricciones)."""
            d = pd.DataFrame(h.filas)
            d["J_real"], d["viol_real"] = zip(*[real(r) for r in d[TH].to_numpy(float)])
            Jr, vr = real(th)
            conv.append(d)
            corr.append(pd.DataFrame([{"formulacion": met, "corrida": k, "modo_ref": "L", "t_total_s": t_total,
                                       "iteraciones": len(h.filas) - 1, **dict(zip(TH, th)), **extra,
                                       "J_real": Jr, "viol_real": vr}]))
            for ruta, filas in ((f_conv, conv), (f_corr, corr)):
                tmp = ruta.with_suffix(".tmp")
                pd.concat(filas).to_csv(tmp, index=False)
                tmp.replace(ruta)
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
            if len(hechas.intersection(range(len(ths)))) == len(ths):
                return
            cfg = {"F4": dict(metodo="trapecio"), "F4b": dict(metodo="trapecio", hess="lbfgs", max_iter=500),
                   "F5": dict(metodo="disparo", max_iter=250)}[met]
            P = form_directa.ProblemaDirecto(p, c, [esc[0]], modo="L", **cfg)
            log(f"{met}: NLP construido en {P.t_construccion:.1f} s ({P.nw} variables, {P.ng} restricciones)")
            for k, th0 in enumerate(ths):
                if k in hechas:
                    continue
                h = Historial(met, k, e0)
                r = P.resolver(th0, h, cota_V=cv)
                cerrar(h, k, r["th"], r["t_inicial_s"] + r["t_s"], t_solver_s=r["t_s"], t_inicial_s=r["t_inicial_s"],
                       t_construccion_s=r["t_construccion_s"], n_var=r["n_var"], n_res=r["n_res"],
                       estado=r["estado"], exito=r["exito"], J_modelo=r["J"], zeta=r["zeta"][0],
                       n_cortes=r["n_cortes"], lam_zeta=r["lam_zeta"], t_limite_s=r["t_limite_s"])


# ============================================================ restricciones frente a limitador
def exp_modos():
    ths = semillas()
    filas = []
    with OraculoJulia(WORKERS) as orac:
        for modo, k, cv in (("R", 2, None), ("L", 2, None), ("L", 0, cota("L_S1", [0]))):
            nombre = esc[k]["nombre"]
            pr = problema(orac, [k], modo, cota_V=cv)
            corr, mejor = multiarranque(pr, ths, f"{nombre} modo {modo}")
            Js = np.array([f["J"] for f in corr if f["viol"] <= TOL_FACT])
            filas.append(fila_sintonia(pr, mejor["z"], escenario=nombre, modo=modo, metodo="F2", n_factibles=len(Js),
                                       n_mejor=int(np.sum(Js <= mejor["J"] * (1 + 1e-3))),
                                       J_cuencas=" ".join(f"{v:.4f}" for v in sorted(set(np.round(Js, 4))))))
            P = form_directa.ProblemaDirecto(p, c, [esc[k]], "trapecio", modo=modo)
            r = P.resolver(pr.theta(mejor["z"]), cota_V=cv)
            z = pr.z_de(r["th"])
            filas.append(fila_sintonia(pr, z, escenario=nombre, modo=modo, metodo="F4", estado=r["estado"],
                                       t_s=r["t_s"], t_limite_s=r["t_limite_s"], n_cortes=r["n_cortes"],
                                       J_modelo=r["J"], **{k: v for k, v in r.items() if k.startswith("lam_")
                                                           and not isinstance(v, list)},
                                       lam_est=" ".join(f"{v:.3e}" for v in r["lam_est"]),
                                       lam_R9=" ".join(f"{v:.3e}" for v in r["lam_R9"])))
            log(f"modo {modo} F4: {r['estado']} J = {r['J']:.5f} (Julia {filas[-1]['J']:.5f})")
            pd.DataFrame(filas).to_csv(R / "modos.csv", index=False)


# ============================================================ ablación de restricciones
def exp_ablacion():
    ths = semillas()
    cv = cota("L_S1", [0])
    casos = [("Artículo: cotas + P ≤ Pmax", dict(grupos=(), estaticas=(), p_max=True)),
             ("+ R2 frecuencia y R6 tensión", dict(grupos=("R2", "R6"), estaticas=(), p_max=True)),
             ("+ R3 corriente", dict(grupos=("R2", "R6", "R3"), estaticas=(), p_max=True)),
             ("+ R4 límites del BESS", dict(grupos=("R2", "R6", "R3", "R4"), estaticas=(), p_max=True)),
             ("+ R5 energía de servicio", dict(grupos=("R2", "R6", "R3", "R4", "R5"), estaticas=(), p_max=True)),
             ("+ R7-R8 inercia", dict(grupos=("R2", "R6", "R3", "R4", "R5"), estaticas=("R7", "R8"), p_max=True)),
             ("+ R9 control de tensión", dict(estaticas=("R7", "R8", "R9"), p_max=True)),
             ("+ R11 validez fasorial", dict(p_max=True)),
             ("+ ζ ≥ 10 %", dict(p_max=True, zeta_min=p["optimizacion"]["zeta_min"])),
             ("limitador en la dinámica (L), con ζ", dict(modo="L", zeta_min=p["optimizacion"]["zeta_min"])),
             ("(P): + calidad de tensión", dict(modo="L", zeta_min=p["optimizacion"]["zeta_min"], cota_V=cv))]
    filas, previa = [], []
    with OraculoJulia(WORKERS) as orac:
        completo = problema(orac, [0], cota_V=cv)
        _, sol_P = multiarranque(completo, ths, "(P) completo")
        th_P = completo.theta(sol_P["z"])
        for nombre, kw in casos:
            pr = problema(orac, [0], **{"modo": "R", "zeta_min": 0, **kw})
            arranques = ths + previa + [th_P]
            _, mejor = multiarranque(pr, arranques, nombre)
            z, fijan = mejor["z"], ""
            if mejor["viol"] > TOL_FACT:
                fase = fase_uno(pr, arranques + [pr.theta(z)])
                z, fijan = fase["z"], fase["fijan"]
                log(f"{nombre}: fase I t* = {fase['t_estrella']:.5g}")
            previa = [pr.theta(z)]
            f = fila_sintonia(pr, z, caso=nombre, modo="L" if pr.lim else "R", fijan=fijan)
            f["viol_P"] = completo.viol(completo.z_de(pr.theta(z)))
            filas.append(f)
            pd.DataFrame(filas).to_csv(R / "ablacion.csv", index=False)


# ============================================================ sensibilidad del objetivo
def exp_pareto():
    ths, cv = semillas(), cota("L_S1", [0])
    o, filas, previa = p["optimizacion"], [], []
    with OraculoJulia(WORKERS) as orac:
        for wl in (0.0, 0.1, 1.0):
            wJ = (o["wf"], o["wv"], wl)
            pr = problema(orac, [0], cota_V=cv, pesos_J=wJ)
            _, mejor = multiarranque(pr, ths + previa, f"wl={wl}")
            P = form_directa.ProblemaDirecto(p, c, [esc[0]], modo="L", pesos_J=wJ)
            base = P.resolver(pr.theta(mejor["z"]), cota_V=cv)
            previa = [base["th"]]
            f = fila_sintonia(pr, pr.z_de(base["th"]), factor_cota=1.0, wl=wl, lam_V=base["lam_V"],
                              J_modelo=base["J"], estado=base["estado"], t_limite_s=base["t_limite_s"],
                              n_cortes=base["n_cortes"], lam_zeta=base["lam_zeta"])
            filas.append(f)
            if wl == 0.1 and cv is not None:
                laterales = []
                for factor in (0.98, 1.02):
                    lat = P.resolver(base["th"], cota_V=factor * cv)
                    ref = problema(orac, [0], cota_V=factor * cv, pesos_J=wJ)
                    filas.append(fila_sintonia(ref, ref.z_de(lat["th"]), factor_cota=factor, wl=wl,
                                               lam_V=lat["lam_V"], J_modelo=lat["J"], estado=lat["estado"],
                                               t_limite_s=lat["t_limite_s"], n_cortes=lat["n_cortes"],
                                               lam_zeta=lat["lam_zeta"]))
                    laterales.append(lat)
                ok = all(r["exito"] for r in [base, *laterales]) and all(q["viol"] <= TOL_FACT for q in filas[-3:])
                dif = (laterales[1]["J"] - laterales[0]["J"]) / (0.04 * cv[0]) if ok and cv[0] > 0 else np.nan
                pred = -base["lam_V"] if ok else np.nan
                f.update(dJ_dcota_multiplicador=pred, dJ_dcota_diferencias=dif,
                         error_relativo=abs(pred - dif) / abs(dif) if abs(dif) >= 1e-3 else np.nan,
                         mismo_activo=len({q["activas"] for q in filas[-3:]}) == 1)
            pd.DataFrame(filas).to_csv(R / "pareto.csv", index=False)


# ============================================================ barridos
def exp_barridos():
    ths, cv = semillas(), cota("L_S1", [0])
    r0 = p["bess"]["reserva"]
    filas = []
    P = form_directa.ProblemaDirecto(p, c, [esc[0]], "trapecio", modo="L")
    with OraculoJulia(WORKERS) as orac:
        # Energía reservada: multiarranque con SQP (más la solución del punto anterior, continuación) y luego
        # colocación desde esa solución, que entrega los multiplicadores. El multiplicador se contrasta con la
        # pendiente por diferencias centradas (±1 %), resueltas con la colocación desde la misma solución.
        previa = []
        for rho in (0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.10):
            pr = problema(orac, [0], reserva=rho, cota_V=cv)
            _, mejor = multiarranque(pr, ths + previa, f"reserva {rho}")
            base = P.resolver(pr.theta(mejor["z"]), reserva=rho, cota_V=cv)
            previa = [base["th"]]
            Jc = [P.resolver(base["th"], reserva=rho * fr, cota_V=cv) for fr in (0.99, 1.01)]
            ok = base["exito"] and all(r["exito"] for r in Jc) and pr.viol(pr.z_de(base["th"])) <= TOL_FACT
            act = [activas(pr, pr.z_de(base["th"]))]
            for fr, res in zip((0.99, 1.01), Jc):
                ref = problema(orac, [0], reserva=rho * fr, cota_V=cv)
                z = ref.z_de(res["th"])
                act.append(activas(ref, z))
                ok &= ref.viol(z) <= TOL_FACT
            dif = (Jc[1]["J"] - Jc[0]["J"]) / (0.02 * rho) if ok else np.nan
            # dJ*/dρ = -Σ_i λ_i dE_res,i/dρ, con λ de las filas de energía de servicio y de R8a
            dE = c["Eres"] / r0
            pred = -(base["lam_E0"] * dE[0] + base["lam_E1"] * dE[1]
                     + base["lam_est"][1] * dE[0] + base["lam_est"][2] * dE[1]) if ok else np.nan
            filas.append(fila_sintonia(pr, pr.z_de(base["th"]), barrido="reserva", valor=rho, estado=base["estado"],
                                       J_modelo=base["J"], lam_E1=base["lam_E0"], lam_E2=base["lam_E1"],
                                       t_limite_s=base["t_limite_s"], n_cortes=base["n_cortes"],
                                       lam_zeta=base["lam_zeta"],
                                       dJ_drho_multiplicador=pred,
                                       dJ_drho_diferencias=dif, mismo_activo=len(set(act)) == 1,
                                       error_relativo=abs(pred - dif) / abs(dif) if abs(dif) >= 1e-3 else np.nan))
            log(f"reserva {rho}: J = {base['J']:.5f}, dJ/dρ multiplicador {pred:.4f}, diferencias "
                f"{filas[-1]['dJ_drho_diferencias']:.4f}")
            pd.DataFrame(filas).to_csv(R / "barridos.csv", index=False)
        # Tamaño de la contingencia (el requisito de inercia R7 sigue fijado por los 50 MW de diseño)
        escenarios = []
        for dP in (20.0, 30.0, 40.0, 50.0, 60.0):
            k = orac.nuevo(5, dP)
            escenarios.append((dP, k, orac.red(k, esc[0])))
        for modo in ("L", "R"):
            previa = []
            for dP, k, e in escenarios:
                cota_V = cv * (dP / 50) ** 2 if modo == "L" and cv is not None else None
                pr = fs.ProblemaCN(orac, p, c, [e], [k], ab, modo=modo, cota_V=cota_V)
                _, mejor = multiarranque(pr, ths + previa, f"dP {dP} modo {modo}")
                previa = [pr.theta(mejor["z"])]
                filas.append(fila_sintonia(pr, mejor["z"], barrido=f"dP_{modo}", valor=dP))
                pd.DataFrame(filas).to_csv(R / "barridos.csv", index=False)


# ============================================================ una sintonía para varios escenarios
def exp_robusto():
    ths, conjunta = semillas(), semilla_conjunta()
    ks = PROBLEMAS["L_todos"]
    pi = [e["prob"] for e in p["escenario"]]
    cv_L, cv_R = cota("L_todos", ks), cota("R_S3", [2])
    filas, sem = [], {}
    with OraculoJulia(WORKERS) as orac:
        casos = [(f"opt_S{k+1}", "L", [k], {}) for k in ks]
        casos += [("opt_R_S3", "R", [2], {}),
                  ("minmax", "L", ks, dict(agregacion="minmax")),
                  ("esperado", "L", ks, dict(agregacion="esperado", pesos=pi)),
                  ("cvar80", "L", ks, dict(agregacion="cvar", pesos=pi, alpha=0.8)),
                  ("S1_S3", "L", [0, 1, 2], dict(agregacion="minmax")),
                  ("dos_juegos", "L", [0, 1, 3], dict(agregacion="minmax")),
                  ("dos_juegos", "L", [2], {})]
        for nombre, modo, diseno, kw in casos:
            cv = cv_R if modo == "R" else None if cv_L is None else cv_L[diseno]
            margen = 0.005 if len(diseno) > 1 else 0.0
            pr = problema(orac, diseno, modo, cota_V=cv, margen=margen, **kw)
            arranques = ths + (conjunta + [sem[k] for k in diseno if k in sem] if modo == "L" else [])
            corr, mejor = multiarranque(pr, arranques, f"{nombre} {modo}")
            z = mejor["z"]
            th = pr.theta(z)
            if len(diseno) == 1 and nombre.startswith("opt_S") and mejor["viol"] <= TOL_FACT:
                sem[diseno[0]] = th
            juego = ("A" if len(diseno) > 1 else "B") if nombre == "dos_juegos" else ""
            f = fila_sintonia(pr, z, modo=modo, sintonia=nombre, juego=juego,
                              criterio=kw.get("agregacion", "un escenario"),
                              escenarios=" ".join(f"S{k+1}" for k in diseno),
                              viol_min_corridas=min(r["viol"] for r in corr))
            Js = []
            for k in ks:
                cv = cv_R if modo == "R" and k == 2 else cv_L[k:k+1] if modo == "L" and cv_L is not None else None
                ref = problema(orac, [k], modo, margen=margen, cota_V=cv)
                zref = ref.z_de(th)
                Js.append(ref.J_real(zref))
                f.update({f"J_S{k+1}": Js[-1], f"viol_S{k+1}": ref.viol(zref), f"zeta_S{k+1}": ref.zeta(zref)})
            f.update(J_max=max(Js), J_esp=float(np.dot(pi, Js)))
            filas.append(f)
            tmp = R / "robusto.csv.tmp"
            pd.DataFrame(filas).to_csv(tmp, index=False)
            tmp.replace(R / "robusto.csv")


# ============================================================ refinamiento de malla
def exp_malla():
    mod = pd.read_csv(R / "modos.csv")
    th0 = mod[(mod.escenario == esc[0]["nombre"]) & (mod.modo == "L") &
              (mod.metodo == "F4")].iloc[0][TH].to_numpy(float)
    cv = cota("L_S1", [0])
    f_malla = R / "malla.csv"
    prev = (pd.read_csv(f_malla)
            if f_malla.exists() and f_malla.stat().st_mtime >= (R / "modos.csv").stat().st_mtime
            else pd.DataFrame())
    filas = prev.to_dict("records") if not prev.empty else []
    hechos = {int(f) for f in prev["factor"]} if not prev.empty else set()
    with OraculoJulia(1) as orac:
        ref = problema(orac, [0], cota_V=cv)
        for factor in (1, 2, 4):
            if factor in hechos:
                continue
            P = form_directa.ProblemaDirecto(p, c, [esc[0]], "trapecio", modo="L", factor=factor)
            r = P.resolver(th0, cota_V=cv)
            z = ref.z_de(r["th"])
            filas.append({"factor": factor, "nodos": len(P.t), "n_var": r["n_var"], "estado": r["estado"],
                          "t_s": r["t_s"], "t_construccion_s": r["t_construccion_s"], "J_modelo": r["J"],
                          "zeta": r["zeta"][0], "n_cortes": r["n_cortes"], "lam_zeta": r["lam_zeta"],
                          "t_limite_s": r["t_limite_s"],
                          "J_real": ref.J_real(z), "viol_real": ref.viol(z), **dict(zip(TH, r["th"]))})
            log(f"malla ×{factor}: {r['estado']} J = {r['J']:.6f}, Julia {filas[-1]['J_real']:.6f}, "
                f"violación {filas[-1]['viol_real']:.1e}, {r['t_s']:.0f} s")
            pd.DataFrame(filas).to_csv(f_malla, index=False)


# ============================================================ sintonías finales
def exp_optimos():
    filas = [{"sintonia": "inicial", "modo": "R", "escenario": esc[0]["nombre"],
              **dict(zip(TH, p["optimizacion"]["theta_inicial"]))}]
    def agregar(d, nombre, modo, k=0, J="J", viol="viol"):
        fact = d[d[viol] <= TOL_FACT].sort_values(J)
        if fact.empty:
            log(f"{nombre}: se omite porque no tiene corrida factible")
            return
        f = fact.iloc[0]
        filas.append({"sintonia": nombre, "modo": modo, "escenario": esc[k]["nombre"], **{n: f[n] for n in TH}})

    f1 = pd.read_csv(R / "corridas_F1.csv")
    agregar(f1, "F1_articulo", "R", J="J_real", viol="viol_real")
    mod = pd.read_csv(R / "modos.csv")
    rob = pd.read_csv(R / "robusto.csv")
    agregar(rob[(rob.sintonia == "opt_R_S3") & (rob.modo == "R")], "opt_R_S3", "R", k=2)
    agregar(mod[(mod.escenario == esc[0]["nombre"]) & (mod.modo == "L")], "opt_L", "L")
    for s in ("minmax", "esperado", "cvar80", "S1_S3"):
        agregar(rob[(rob.sintonia == s) & (rob.modo == "L")], f"{s}_L", "L")
    for juego, k in (("A", 0), ("B", 2)):
        agregar(rob[(rob.sintonia == "dos_juegos") & (rob.juego == juego)], f"dos_{juego}", "L", k=k)
    pd.DataFrame(filas).to_csv(R / "optimos.csv", index=False)
    log(f"optimos.csv con {len(filas)} sintonías")


# ============================================================ certificados
def _cert_uno(tarea):
    from gfm import certificado
    c_loc, esc_loc, th_names, r = tarea
    As, fila = [], {"sintonia": r["sintonia"], "modo": r["modo"]}
    th = np.array([r[n] for n in th_names], float)
    for e in esc_loc:
        A, resid = certificado.linealizar(c_loc, {**e, "lim": float(r["modo"] == "L")}, th)
        ev = np.linalg.eigvals(A)
        fila[f"zeta_{e['nombre']}"] = certificado.zeta_polos(ev, resid)
        fila[f"alpha_{e['nombre']}"] = float(-ev.real.max())
        fila[f"residuo_{e['nombre']}"] = resid
        As.append(A)
    fila["zeta_min"] = min(fila[f"zeta_{e['nombre']}"] for e in esc_loc)
    fila["alpha_eig_min"] = min(fila[f"alpha_{e['nombre']}"] for e in esc_loc)
    try:
        fila["alpha_LMI_comun"] = certificado.tasa_certificada_comun(As)
    except AssertionError as e:                            # SDP sin decisión numérica: no es prueba de inexistencia
        fila["alpha_LMI_comun"], fila["lmi_nota"] = np.nan, str(e)
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
    ap.add_argument("--problema", choices=PROBLEMAS)
    ap.add_argument("--caso", choices=SENS)
    ap.add_argument("--res", default=None, help="carpeta de resultados (por defecto resultados/)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("GFM_WORKERS", "0")),
                    help="procesos de Julia en paralelo (0 = mitad de hilos de CPU)")
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n debe ser positivo")
    if args.exp == "rango" and args.problema is None:
        ap.error("rango requiere --problema")
    if args.exp == "sens" and args.caso is None:
        ap.error("sens requiere --caso")
    WORKERS = max(1, (os.cpu_count() or 2) // 2) if args.workers <= 0 else args.workers
    R = datos.RESULTADOS if args.res is None else Path(args.res)
    R.mkdir(exist_ok=True, parents=True)
    if args.exp == "sens":
        preparar_sens(args.caso, R / "sens_datos" / args.caso)
    p, c, esc = datos.cargar()
    ab = modelo.sensibilidad_tension(c, esc[0])
    {"conv": exp_conv, "modos": exp_modos, "ablacion": exp_ablacion, "pareto": exp_pareto, "barridos": exp_barridos,
     "robusto": exp_robusto, "malla": exp_malla, "optimos": exp_optimos, "cert": exp_cert,
     "rango": exp_rango, "kappa": exp_kappa, "fase1": exp_fase1, "sens": exp_sens}[args.exp]()
    log(f"Listo: {args.exp}")


if __name__ == "__main__":
    main()
