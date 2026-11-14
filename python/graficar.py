"""Figuras del informe final; ejecutar de nuevo al terminar los experimentos."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

R = Path(__file__).resolve().parents[1] / "resultados"
SAL = Path(sys.argv[1]) if len(sys.argv) > 1 else R.parents[1] / "Informe" / "figuras"
SAL.mkdir(parents=True, exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False,
                     "savefig.bbox": "tight", "font.family": "serif"})
ETIQ = {"F1": "F1 (artículo) (SQP+PSO)", "F2": "F2 SQP", "F3": "F3 PSO",
        "F4": "F4 colocación", "F4b": "F4b colocación L-BFGS", "F5": "F5 disparo múltiple"}


def leer(nombre):
    try:
        return pd.read_csv(R / nombre)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Aviso: {nombre} ausente o vacío; datos pendientes.")
        return pd.DataFrame()


def pendiente(nombre):
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    ax.axis("off")
    ax.text(0.5, 0.5, "pendiente", ha="center", va="center", transform=ax.transAxes)
    for eje in fig.axes:
        if eje.get_xscale() == "linear":
            eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    fig.savefig(SAL / nombre, bbox_inches=fig.bbox_inches)
    plt.close(fig)


tr = leer("trayectorias.csv")
if tr.empty:
    pendiente("trayectorias_final.pdf")
else:
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 4.2), sharex=True)
    for k, (s, etiqueta) in enumerate([("inicial", "θ₀"), ("F1_articulo", "F1 (artículo)"),
                                         ("opt_R", "óptimo R"), ("opt_L", "óptimo L")]):
        for lim in ([1, 0] if s == "F1_articulo" else [1]):
            g = tr[(tr.sintonia == s) & (tr.lim == lim)].sort_values("t")
            if g.empty:
                continue
            for ax, col in zip(axs, ["f_sg", "I1", "pb1"]):
                ax.plot(g.t, g[col], color=SERIES[k], ls="-" if lim else "--",
                        label=etiqueta if lim else "F1 (artículo) sin limitador")
    for ax, etiqueta in zip(axs, [r"$f_{SG}$ [Hz]", r"$I_1$ [pu]", r"$p_{b,1}$ [pu]"]):
        ax.set_ylabel(etiqueta)
        ax.set_xlim(0, 30)
    for ax in axs[1:]:
        ax.axhline(1.0, color="#52514e", lw=0.8, ls=":")
    axs[1].axhline(1.3, color="#52514e", lw=0.8, ls=":")
    axs[-1].set_xlabel("Tiempo [s]")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=0.5)
    fig.savefig(SAL / "trayectorias_final.pdf", bbox_inches=fig.bbox_inches)
    plt.close(fig)

corridas = {p.stem.removeprefix("corridas_"): leer(p.name) for p in sorted(R.glob("corridas_F*.csv"))}
finales = pd.concat(corridas.values(), ignore_index=True) if corridas else pd.DataFrame()
Jstar = finales.loc[finales.viol_real <= 1e-3, "J_real"].min() if not finales.empty else np.nan
if not np.isfinite(Jstar):
    print("Aviso: no hay J* factible; perfil de convergencia pendiente.")
    pendiente("convergencia_final.pdf")
else:
    print(f"J* global factible = {Jstar:.10g}; umbral del perfil = {1.01 * Jstar:.10g}")
    perfiles = []
    for k, (f, etiqueta) in enumerate(ETIQ.items()):
        c = corridas.get(f, pd.DataFrame())
        if c.empty:
            print(f"Aviso: corridas_{f}.csv ausente o vacío; se omite {f}.")
            continue
        g = leer(f"conv_{f}.csv")
        if g.empty:
            continue
        g = g[g.corrida.isin(c.corrida.unique()) & np.isfinite(g.t_s) & (g.t_s >= 0)]
        exitos = g[(g.viol_real <= 1e-3) & (g.J_real <= 1.01 * Jstar)]
        tiempos = exitos.groupby("corrida").t_s.min().sort_values().to_numpy()
        perfiles.append((k, etiqueta, c.corrida.nunique(), tiempos, g.t_s.to_numpy()))
    if not perfiles:
        pendiente("convergencia_final.pdf")
    else:
        observados = np.concatenate([p[4] for p in perfiles])
        positivos = observados[observados > 0]
        inicio = min(0.01, positivos.min() / 2) if positivos.size else 0.01
        fin = max(1.0, observados.max() * 1.05) if observados.size else 1.0
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        for k, etiqueta, n, tiempos, _ in perfiles:
            x = np.r_[inicio, np.maximum(tiempos, inicio), fin]
            y = np.r_[0, np.arange(1, len(tiempos) + 1) / n, len(tiempos) / n]
            ax.step(x, y, where="post", color=SERIES[k], label=etiqueta)
        ax.set(xscale="log", xlim=(inicio, fin), ylim=(0, 1.02),
               xlabel="Tiempo de cómputo [s]", ylabel="Fracción de corridas")
        ax.legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        for eje in fig.axes:
            if eje.get_xscale() == "linear":
                eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
            eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        fig.savefig(SAL / "convergencia_final.pdf", bbox_inches=fig.bbox_inches)
        plt.close(fig)

b = leer("barridos.csv")
if b.empty:
    pendiente("barridos_final.pdf")
else:
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.0))
    g = b[(b.barrido == "reserva") & (b.viol <= 1e-3)].sort_values("valor")
    if not g.empty:
        x = 100 * g.valor.to_numpy()
        axs[0].plot(x, g.J_modelo, "o-", color=SERIES[0], ms=3)
        distintos = np.unique(x)
        semiancho = np.diff(distintos).min() * 0.18 if len(distintos) > 1 else 0.2
        for _, r in g.iterrows():
            dx = np.array([-semiancho, semiancho])
            axs[0].plot(100 * r.valor + dx, r.J_modelo + dx * r.dJ_drho_multiplicador / 100,
                        color=SERIES[1], lw=1)
    else:
        axs[0].text(0.5, 0.5, "pendiente", ha="center", transform=axs[0].transAxes)
    for k, modo in enumerate(["R", "L"]):
        g = b[b.barrido == f"dP_{modo}"].sort_values("valor")
        if not g.empty:
            axs[1].plot(g.valor, g.dfmax_Hz, "o-", color=SERIES[k], ms=3, label=f"Modo {modo}")
            malas = g[g.viol > 1e-3]
            axs[1].plot(malas.valor, malas.dfmax_Hz, "x", color="#b11b1b", ms=6, mew=1.5)
    if axs[1].lines:
        axs[1].legend(fontsize=6)
    else:
        axs[1].text(0.5, 0.5, "pendiente", ha="center", transform=axs[1].transAxes)
    axs[0].set(xlabel="Energía reservada [%]", ylabel=r"$J_{modelo}$")
    axs[1].set(xlabel="Escalón [MW]", ylabel=r"$\Delta f_{max}$ [Hz]")
    fig.tight_layout(w_pad=0.8)
    for eje in fig.axes:
        eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    fig.savefig(SAL / "barridos_final.pdf", bbox_inches=fig.bbox_inches)
    plt.close(fig)
print("Figuras en", SAL)
