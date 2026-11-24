"""Carga de parámetros (datos/parametros.toml) y de la red exportada por Julia."""
import subprocess
import tomllib
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]          # carpeta Codigos_final/
RESULTADOS = RAIZ / "resultados"
NOMBRES_THETA = ["H1", "H2", "D1", "D2", "Kq1", "Kq2", "K1", "K2"]
NOMBRES_EST = ["R7", "R8a_1", "R8a_2", "R8b_1", "R8b_2", "R11_1", "R11_2"]


def _leer(ruta):
    with open(ruta, "rb") as f:
        return tomllib.load(f)


def cargar():
    """Devuelve (p, c, escenarios): parámetros, constantes del punto de operación y red por escenario."""
    red = RAIZ / "datos" / "red_exportada.toml"
    if not red.exists() or red.stat().st_mtime < (RAIZ / "datos" / "parametros.toml").stat().st_mtime:
        subprocess.run(["julia", "--project=julia", str(Path("julia") / "exportar_red.jl")], cwd=RAIZ, check=True)
    p, r = _leer(RAIZ / "datos" / "parametros.toml"), _leer(red)
    c = {k: np.asarray(v) if isinstance(v, list) else v for k, v in r["constantes"].items()}
    esc = []
    for e in r["escenario"]:
        esc.append({"nombre": e["nombre"], "dP": e["dP"], "x0": np.asarray(e["x0"]),
                    "GL": np.asarray(e["GL"]), "pfv": np.asarray(e["pfv"]), "pb0": np.asarray(e["pb0"]),
                    "Y": np.asarray(e["Yred_re"]) + 1j * np.asarray(e["Yred_im"]),
                    "W": np.asarray(e["W_re"]) + 1j * np.asarray(e["W_im"]),
                    "Z": np.asarray(e["Z_re"]) + 1j * np.asarray(e["Z_im"]),
                    "Sc": np.asarray(e["Sc_re"]) + 1j * np.asarray(e["Sc_im"]), "lim": 1.0})
    return p, c, esc


def malla(p, factor=1):
    """Malla de tiempos no uniforme de parametros.toml (idéntica a la de Julia). factor > 1 la refina."""
    t = [0.0]
    for tf, h in p["optimizacion"]["malla"]:
        n = int(round((tf - t[-1]) / h)) * factor
        t += list(np.linspace(t[-1], tf, n + 1)[1:])
    return np.array(t)


def limites(p):
    o = p["optimizacion"]
    lb = np.array([o["H_lim"][0]] * 2 + [o["D_lim"][0]] * 2 + [o["Kq_lim"][0]] * 2 + [o["K_lim"][0]] * 2)
    ub = np.array([o["H_lim"][1]] * 2 + [o["D_lim"][1]] * 2 + [o["Kq_lim"][1]] * 2 + [o["K_lim"][1]] * 2)
    return lb, ub


def holgura_potencia(p, c, escs):
    """Potencia [MW] que cada GFM puede agregar sobre su punto previo: el menor entre la holgura del BESS y la
    de corriente a I_max, en el peor de los escenarios dados."""
    dP = np.full(2, np.inf)
    for e in escs:
        bess = (1 - e["pb0"]) * c["Pn"]
        corr = (np.sqrt((c["Imax"] * c["U0"]) ** 2 - c["q0"] ** 2) - c["p0"]) * c["S"]
        dP = np.minimum(dP, np.minimum(bess, corr))
    return dP


def restricciones_estaticas(p, c, escs, reserva=None):
    """Restricciones lineales G·θ ≤ h sobre inercia y separación de escalas:
    R7   inercia mínima del sistema para la contingencia de diseño: Hg·Sg + Σ Hi·Si ≥ f0·ΔP/(2·RoCoF_max)
    R8a  energía de la respuesta inercial respaldada por la reserva: 2·Hi·Si·Δf_max/f0 ≤ E_res,i  (Hu et al.)
    R8b  inercia efectiva (AT-IBR Art. 7-6 k): la potencia inercial a RoCoF = 2 Hz/s cabe en la holgura,
         2·Hi·Si·RoCoF_max/f0 ≤ ΔP_disp,i
    R11  validez fasorial: 2·Hi/Di ≥ n_sep·max(τ_q)
    """
    n, S, f0 = p["norma"], c["S"], c["f0"]
    Eres = c["Eres"] * (1.0 if reserva is None else reserva / p["bess"]["reserva"])
    req = f0 * n["dP_diseno"] / (2 * n["rocof_max"]) - c["Hg"] * c["Sg"]
    G = np.zeros((7, 8))
    G[0, :2] = -S
    for i in range(2):
        G[1 + i, i] = 2 * S[i] * (f0 - n["f_min_perm"]) / f0
        G[3 + i, i] = 2 * S[i] * n["rocof_max"] / f0
        G[5 + i, i] = -2
        G[5 + i, 2 + i] = p["gfm"]["n_sep"] * max(c["tau_q"])
    h = np.concatenate([[-req], Eres, holgura_potencia(p, c, escs), [0.0, 0.0]])
    return G, h


def puntos_iniciales(p, n, semilla=2026):
    """θ inicial del archivo de parámetros + (n-1) puntos por hipercubo latino."""
    lb, ub = limites(p)
    rng = np.random.default_rng(semilla)
    m = n - 1
    if m == 0:
        return [np.array(p["optimizacion"]["theta_inicial"], float)]
    u = (rng.permuted(np.tile(np.arange(m), (8, 1)), axis=1).T + rng.random((m, 8))) / m
    return [np.array(p["optimizacion"]["theta_inicial"], float)] + list(lb + u * (ub - lb))


def cotas_tension(p, c):
    """R9: cotas de la rapidez del lazo Q-V, r_i = K_i (Kq_i α_i + β_i) = 1/τ_v.
    Inferior: tiempo de crecimiento 2,2·τ_v ≤ t_crec,max (AT-IBR Art. 3-3 e).
    Superior: separación de escalas con el filtro de Q, τ_v ≥ n_sep·τ_q (validez de la aproximación de 1er orden)."""
    return 2.2 / p["norma"]["t_crec_tension"], 1.0 / (p["gfm"]["n_sep"] * float(np.max(c["tau_q"])))


def g_tension(th, ab, p, c):
    """R9 como g ≤ 0 (cuatro componentes: cota inferior y superior de cada GFM)."""
    a, b = ab
    lo, hi = cotas_tension(p, c)
    r = np.array([th[6 + i] * (th[4 + i] * a[i] + b[i]) for i in range(2)])
    return np.concatenate([lo - r, r - hi])
