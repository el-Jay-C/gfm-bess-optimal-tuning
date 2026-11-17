"""Comprueba R11, la referencia ε y la solución directa en S1/R."""
import numpy as np
from gfm import datos, modelo
from gfm.form_directa import ProblemaDirecto
from gfm.form_simulacion import ProblemaCN
from gfm.oraculo import OraculoJulia

p, c, esc = datos.cargar()
th0 = np.array(p['optimizacion']['theta_inicial'], float)
G, h = datos.restricciones_estaticas(p, c, esc[:1])
assert G.shape == (7, 8) and h.shape == (7,)
assert np.all(G[5:] @ th0 <= h[5:])
with OraculoJulia(1) as orac:
    pr = ProblemaCN(orac, p, c, esc[:1], [0], modelo.sensibilidad_tension(c, esc[0]))
    g = pr.g(pr.z_de(th0))
    assert len(g) == len(pr.cols) + 1 + 7 + 4
    assert abs(g[len(pr.cols)]) < 1e-6
P = ProblemaDirecto(p, c, esc[:1], metodo='trapecio', modo='R')
r = P.resolver(th0)
assert r['estado'] == 'Solve_Succeeded', r['estado']
assert r['vdev'][0] <= r['eps'][0] + 1e-6
print('Formulación v3 OK:', r['estado'], 'vdev =', r['vdev'][0], 'eps =', r['eps'][0])
