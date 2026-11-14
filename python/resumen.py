"""Tablas booktabs y números para la redacción del informe final."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "resultados"
SAL = Path(sys.argv[1]) if len(sys.argv) > 1 else R.parents[1] / "Informe" / "tablas"
SAL.mkdir(parents=True, exist_ok=True)
NOMBRES = {"inicial": r"$\theta$ inicial", "F1_articulo": "F1 artículo", "opt_R": "Óptimo R",
           "opt_L": "Óptimo L", "minmax": "Mín. máximo", "minmax_F4": "Mín. máximo (F4)",
           "esperado": "Valor esperado", "cvar": "CVaR", "cvar80": "CVaR",
           "afin": "Regla afín", "afin_minmax": "Regla afín"}
FORM = {"F1": "F1 artículo (SQP+PSO)", "F2": "F2 SQP", "F3": "F3 PSO",
        "F4": "F4 colocación", "F4b": "F4b colocación L-BFGS", "F5": "F5 disparo múltiple"}
TH = ["H1", "H2", "D1", "D2", "Kq1", "Kq2", "K1", "K2"]
CAB_TH = [r"$H_1$", r"$H_2$", r"$D_1$", r"$D_2$", r"$K_{q,1}$", r"$K_{q,2}$", r"$K_1$", r"$K_2$"]


def leer(nombre):
    try:
        return pd.read_csv(R / nombre)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Aviso: {nombre} ausente o vacío; pendiente.")
        return pd.DataFrame()


def texto(valor):
    return "".join({"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c) for c in str(valor))


def nombre(valor):
    return NOMBRES.get(valor, texto(valor))


def num(valor, dec=3):
    if pd.isna(valor) or not np.isfinite(float(valor)):
        return "--"
    return f"{float(valor):.{dec}f}".replace(".", "{,}")


def tabla(archivo, cabecera, filas):
    n = len(cabecera)
    if not filas:
        filas = [[rf"\multicolumn{{{n}}}{{c}}{{pendiente}}"]]
    contenido = [r"\begin{tabular}{l" + "r" * (n - 1) + "}", r"\toprule",
                 " & ".join(cabecera) + r" \\", r"\midrule"]
    contenido += [" & ".join(fila) + r" \\" for fila in filas]
    contenido += [r"\bottomrule", r"\end{tabular}"]
    if archivo == "sintonias.tex":
        contenido = [r"\resizebox{\textwidth}{!}{%"] + contenido + ["}"]
    (SAL / archivo).write_text("\n".join(contenido) + "\n", encoding="utf-8")


opt = leer("optimos.csv")
val = leer("validacion.csv")
cert = leer("certificados.csv")
filas = []
for _, r in opt.iterrows():
    s = r.sintonia
    v = val[(val.sintonia == s) & (val.escenario == "S1_carga_b5")] if not val.empty else pd.DataFrame()
    con = v[v.lim == 1] if not v.empty else pd.DataFrame()
    sin = v[v.lim == 0] if not v.empty else pd.DataFrame()
    c = cert[cert.sintonia == s] if not cert.empty else pd.DataFrame()
    # El certificado ya contiene el mínimo entre los escenarios.
    filas.append([nombre(s)] + [num(r.get(k, np.nan)) for k in TH]
                 + [num(con.iloc[0].get(k, np.nan)) if not con.empty else "--"
                    for k in ["J", "dfmax_Hz", "I1_fin", "pb1_pico", "pred1_max"]]
                 + [num(sin.iloc[0].get("max_violacion", np.nan)) if not sin.empty else "--",
                    num(c.zeta_min.min()) if not c.empty else "--"])
tabla("sintonias.tex", ["Sintonía"] + CAB_TH + [r"$J$", r"$\Delta f_{max}$ [Hz]",
      r"$I_1(T)$", r"$p_{b,1}^{pico}$", r"$\rho_1^{max}$", "Viol. (R)", r"$\zeta_{min}$"], filas)

a = leer("ablacion.csv")
filas = [[texto(r.caso)] + [num(r.get(k, np.nan)) for k in TH[:6] + ["dfmax_Hz", "J", "viol_P"]]
         for _, r in a.iterrows()]
tabla("ablacion.tex", ["Caso"] + CAB_TH[:6] + [r"$\Delta f_{max}$ [Hz]", "$J$", "Viol. (P)"], filas)

corridas = {p.stem.removeprefix("corridas_"): leer(p.name) for p in sorted(R.glob("corridas_F*.csv"))}
todas = pd.concat(corridas.values(), ignore_index=True) if corridas else pd.DataFrame()
Jstar = todas.loc[todas.viol_real <= 1e-3, "J_real"].min() if not todas.empty else np.nan
print(f"\nJ* global factible: {Jstar:.10g}; mejor cuenca: J <= {1.005 * Jstar:.10g}")
filas = []
for f, etiqueta in FORM.items():
    g = corridas.get(f, pd.DataFrame())
    if g.empty:
        filas.append([texto(etiqueta), "pendiente"] + ["--"] * 6)
        continue
    fact = g[g.viol_real <= 1e-3]
    mejor = (fact.J_real <= 1.005 * Jstar).sum()
    columna = "n_eval" if f in ["F1", "F2", "F3"] else "iteraciones"
    nv = g.n_var.dropna().unique() if "n_var" in g else []
    nvar = "/".join(num(v, 0) for v in sorted(nv)) if len(nv) else "--"
    filas.append([texto(etiqueta), str(len(g)), str(len(fact)), str(mejor), num(fact.J_real.median(), 4),
                  num(g.t_total_s.median(), 1), num(g[columna].median(), 0) if columna in g else "--", nvar])
    print(f"{f}: {len(g)} corridas, {len(fact)} factibles, {mejor} en mejor cuenca; "
          f"mediana J factible={fact.J_real.median():.6g}, t={g.t_total_s.median():.6g} s")
tabla("convergencia.tex", ["Formulación", "Corridas", "Factibles", "Mejor", r"Med. $J$",
      r"Med. $t$ [s]", "Eval./iter.", r"$n_{var}$"], filas)

rob = leer("robusto.csv")
fue = leer("fuera_muestra.csv")
filas = []
for _, r in rob.iterrows():
    g = fue[fue.sintonia == r.sintonia] if not fue.empty else pd.DataFrame()
    pct, media = "--", "--"
    if not g.empty:
        factible = g.factible.astype(str).str.lower().map({"true": 1.0, "false": 0.0,
                                                         "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})
        pct, media = num(100 * factible.mean(), 1), num(g.J.mean())
    filas.append([nombre(r.sintonia)] + [num(r.get(k, np.nan)) for k in ["J_S1", "J_S2", "J_S3", "J_max", "J_esp"]]
                 + [pct, media])
tabla("robusto.tex", ["Sintonía", r"$J_{S1}$", r"$J_{S2}$", r"$J_{S3}$", r"$J_{max}$",
      r"$J_{esp}$", r"Factibles [\%]", r"$\overline J$"], filas)

print("\nRestricciones activas y multiplicadores (modos.csv)")
modos = leer("modos.csv")
if not modos.empty:
    columnas = [c for c in modos if c in ["modo", "metodo", "sintonia", "J", "viol_P", "activas"]
                or c.startswith("lam_")]
    print(modos[columnas].to_string(index=False))
for archivo in ["pesos.csv", "malla.csv", "validacion_densa.csv", "largo_plazo.csv", "escalon_tension.csv"]:
    print(f"\n{archivo}")
    g = leer(archivo)
    if g.empty:
        continue
    if archivo in ["pesos.csv", "malla.csv"]:
        print(g.to_string(index=False))
    else:
        # Extremos y medias por sintonía; conserva el escenario y los diagnósticos textuales.
        for s, grupo in g.groupby("sintonia", sort=False):
            print(f"{s}: {len(grupo)} filas")
            numericas = grupo.select_dtypes(include="number")
            if not numericas.empty:
                print(numericas.agg(["min", "mean", "max"]).to_string())
            for c in grupo.select_dtypes(exclude="number"):
                if c != "sintonia":
                    print(f"  {c}: {', '.join(grupo[c].dropna().astype(str).unique())}")
print("\ncertificados.csv (amortiguamiento y Lyapunov común)")
if not cert.empty:
    print(cert.to_string(index=False))
print("\nTablas en", SAL)
