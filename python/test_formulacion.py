"""Comprueba las restricciones v4, un corte de amortiguamiento y los datos alternativos."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ejecutar_experimentos import preparar_sens
from gfm import datos, modelo
from gfm.form_directa import ProblemaDirecto
from gfm.form_simulacion import ProblemaCN
from gfm.historial import Historial
from gfm.oraculo import OraculoJulia

p, c, esc = datos.cargar()
th0 = np.array(p["optimizacion"]["theta_inicial"], float)
G, h = datos.restricciones_estaticas(p, c, esc[:1])
assert G.shape == (7, 8) and h.shape == (7,)
assert np.all(G[5:] @ th0 <= h[5:])
ab = modelo.sensibilidad_tension(c, esc[0])

with OraculoJulia(1) as orac:
    v0 = orac.evaluar(th0, 0, 1)[0]["vdev"]
    pr = ProblemaCN(orac, p, c, esc[:1], [0], ab, modo="L", cota_V=[v0])
    g = pr.g(pr.z_de(th0))
    assert len(g) == len(pr.nombres_g()) == 31
    assert abs(g[len(pr.cols)]) < 1e-10
    assert "pb1_sost" in [orac.nombres_g[k] for k in pr.cols]
    sin_cotas = ProblemaCN(orac, p, c, esc[:1], [0], ab, modo="L", zeta_min=0)
    assert len(sin_cotas.g(sin_cotas.z_de(th0))) == 29
    todos = ProblemaCN(orac, p, c, esc, range(len(esc)), ab, modo="L", agregacion="minmax", cota_V=[v0] * len(esc))
    assert len(todos.g(todos.z_de(th0))) == 95
    for k, e in enumerate(esc):
        red = orac.red(k, e)
        for campo in ("dP", "x0", "GL", "pfv", "pb0", "Y", "W", "Z", "Sc"):
            np.testing.assert_allclose(red[campo], e[campo], atol=1e-14)
print("Restricciones CN OK: 31 (S1), 29 (sin cota ni ζ), 95 (S1–S4 min–max). Red del oráculo OK.", flush=True)

with TemporaryDirectory(prefix="gfm_sens_") as tmp:
    original = os.environ.get("GFM_DATOS")
    try:
        preparar_sens("klim4", Path(tmp))
        pa, ca, _ = datos.cargar()
        assert pa["gfm"]["k_lim"] == ca["klim"] == 4.0
        assert pa["gfm"]["kp_lim"] == p["gfm"]["kp_lim"]
        assert c["klim"] == p["gfm"]["k_lim"] != ca["klim"]
        print(f"GFM_DATOS OK: Python k_lim = {pa['gfm']['k_lim']}, Julia klim exportado = {ca['klim']}", flush=True)
    finally:
        if original is None:
            os.environ.pop("GFM_DATOS", None)
        else:
            os.environ["GFM_DATOS"] = original

P = ProblemaDirecto(p, c, esc[:1], modo="L", zeta_min=0)
N = len(P.t) - 1
assert P.ng == (modelo.NX + 3 + 1) * N + (6 + 2 * 5 + 5) * (N + 1) + 4 + 2 + 9, P.ng
assert len(P.filas["V"]) == len(P.filas["zeta"]) == 1
assert len(P.filas["pb_sost0"]) == len(P.filas["pb_sost1"]) == 1
base = P.resolver(th0)
assert base["exito"] and base["n_cortes"] == 0 and base["zeta"][0] > 0, base
print(f"F4 libre: g = {P.ng}, ζ = {base['zeta'][0]:.8f}, vdev = {base['vdev'][0]:.8f}", flush=True)
P.zeta_min = 1.05 * base["zeta"][0]
cv = [1.02 * base["vdev"][0]]
hist = Historial("F4", 0, esc[0]["nombre"])
r = P.resolver(base["th"], hist, cota_V=cv)
assert r["n_cortes"] > 0 and r["zeta"][0] >= P.zeta_min * (1 - 2e-3), r
assert r["vdev"][0] <= cv[0] + 1e-6, r
assert np.all(np.diff([f["t_s"] for f in hist.filas]) >= 0)
assert r["t_limite_s"] == 600
print(f"F4 con corte: {r['estado']}, ζ = {r['zeta'][0]:.8f}, objetivo = {P.zeta_min:.8f}, "
      f"vdev = {r['vdev'][0]:.8f}, cota = {cv[0]:.8f}, cortes = {r['n_cortes']}, "
      f"λζ = {r['lam_zeta']:.5g}, tiempo = {r['t_s']:.1f} s")
