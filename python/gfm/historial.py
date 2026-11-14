"""Registro de la convergencia (tiempo, objetivo, violación y parámetros por iteración)."""
import time

from .datos import NOMBRES_THETA


class Historial:
    def __init__(self, formulacion, corrida, escenario):
        self.base = dict(formulacion=formulacion, corrida=corrida, escenario=escenario)
        self.filas, self.t0 = [], time.perf_counter()

    def registrar(self, J, viol, th, n_eval=None):
        self.filas.append({**self.base, "iter": len(self.filas), "t_s": time.perf_counter() - self.t0,
                           "n_eval": n_eval, "J_modelo": float(J), "viol_modelo": float(viol),
                           **dict(zip(NOMBRES_THETA, map(float, th)))})

    @property
    def t(self):
        return time.perf_counter() - self.t0
