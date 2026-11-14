# Verifica que el modelo CasADi y el de Julia producen las mismas trayectorias, con y sin el limitador
# de sobrecarga activo, y que la simulación de caja negra y el oráculo entregan las mismas métricas.
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gfm import datos, modelo
from gfm.oraculo import OraculoJulia

p, c, esc = datos.cargar()
t = datos.malla(p)
dt = np.diff(t)
wtrap = np.concatenate([[dt[0] / 2], (dt[:-1] + dt[1:]) / 2, [dt[-1] / 2]])
casos = {"inicial": np.array(p["optimizacion"]["theta_inicial"], float),
         "limitador activo": np.array([8.0, 8.0, 50.0, 50.0, 20.0, 20.0, 1.0, 1.0])}
with OraculoJulia() as orac:
    for nombre, th in casos.items():
        for k, e in enumerate(esc):
            f, sal = modelo.construir(c, e)
            X = modelo.simular(f, e["x0"], th, t)
            Y = np.asarray(sal.map(X.shape[1])(X, th))
            m, g = orac.evaluar(th, k, 1)
            F = Y[[modelo.IDX[n] for n in ("f_sg", "f1", "f2")]]
            py = {"dfmax_Hz": np.max(np.abs(F - c["f0"])),
                  "I2_pico": np.sqrt(Y[modelo.IDX["I2sq"]].max()),
                  "I2_fin": np.sqrt(Y[modelo.IDX["I2sq"], -1]),
                  "th2_frac": Y[modelo.IDX["th2"]].max() / c["B"],
                  "pred1_max": Y[modelo.IDX["pred1"]].max(),
                  "pred2_max": Y[modelo.IDX["pred2"]].max(),
                  "perdidas": (Y[modelo.IDX["ploss"]] - c["ploss0"]) @ wtrap / (t[-1] * e["dP"])}
            err = {q: abs(v - m[q]) / max(abs(m[q]), 1e-3) for q, v in py.items()}
            peor = max(err, key=err.get)
            print(f"{nombre:17s} {e['nombre']:12s} error relativo máximo = {err[peor]:.1e} ({peor}); "
                  f"pred máx = {m['pred1_max']:.3f}/{m['pred2_max']:.3f}")
            assert err[peor] < 2e-4, f"Modelos inconsistentes en {peor}: {py[peor]} vs {m[peor]}"
print("Modelos Julia y CasADi consistentes.")
