"""Cliente del oráculo de simulación en Julia (julia/servidor.jl) por stdin/stdout."""
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np

from .datos import RAIZ


class OraculoJulia:
    """Evalúa θ con un pool de procesos Julia. Uso: `with OraculoJulia() as o: m, g = o.evaluar(θ, 0)`."""

    def __init__(self, n_workers=None):
        if n_workers is None:
            n_workers = int(os.environ.get("GFM_WORKERS", "1"))
        if n_workers <= 0:
            n_workers = max(1, (os.cpu_count() or 2) // 2)
        self.n_workers = n_workers
        script = str(Path("julia") / "servidor.jl")
        self.procs = [subprocess.Popen(["julia", "--project=julia", script], cwd=RAIZ, text=True,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                      for _ in range(n_workers)]
        self._pool = queue.Queue()
        self._lock = threading.Lock()
        for proc in self.procs:
            linea = proc.stdout.readline()
            while not linea.startswith("LISTO"):
                if not linea:
                    self.cerrar()
                    raise RuntimeError("El servidor de Julia terminó antes de estar listo")
                linea = proc.stdout.readline()
            m, g = linea[len("LISTO"):].split("|")
            self.nombres_m, self.nombres_g = m.split(), g.split()
            self._pool.put(proc)
        self.ig = {n: i for i, n in enumerate(self.nombres_g)}
        self.cache, self.n_eval = {}, 0

    def _pedir(self, proc, texto):
        proc.stdin.write(texto + "\n")
        proc.stdin.flush()
        return proc.stdout.readline()

    def evaluar(self, th, esc=0, lim=1):
        """Devuelve (métricas: dict, g: ndarray) para θ en el escenario esc (índice desde 0).
        lim = 1 simula con el limitador de sobrecarga; lim = 0, sin él."""
        clave = (esc, lim, *np.round(np.asarray(th, float), 12))
        with self._lock:
            if clave in self.cache:
                return self.cache[clave]
        proc = self._pool.get()
        try:
            linea = self._pedir(proc, f"EVAL {esc + 1} {lim} " + " ".join(f"{v:.12g}" for v in th))
        finally:
            self._pool.put(proc)
        v = np.array(linea.split(), float)
        nm = len(self.nombres_m)
        res = (dict(zip(self.nombres_m, v[:nm])), v[nm:])
        with self._lock:
            if clave not in self.cache:
                self.cache[clave] = res
                self.n_eval += 1
            return self.cache[clave]

    def nuevo(self, barra, dP, soc=(0.5, 0.5), pfv2=45.0):
        """Crea un escenario en todos los workers y devuelve su índice (desde 0)."""
        ocupados = [self._pool.get() for _ in range(self.n_workers)]
        try:
            idx = [int(self._pedir(proc, f"NUEVO {barra} {dP} {soc[0]} {soc[1]} {pfv2}")) - 1 for proc in ocupados]
        finally:
            for proc in ocupados:
                self._pool.put(proc)
        return idx[0]

    def red(self, esc, plantilla):
        """Recupera los datos CasADi de un escenario creado en el oráculo."""
        proc = self._pool.get()
        try:
            v = np.fromstring(self._pedir(proc, f"RED {esc + 1}"), sep=" ")
        finally:
            self._pool.put(proc)
        e, i = {**plantilla, "dP": v[0]}, 1
        for k in ("x0", "GL", "pfv", "pb0"):
            j = i + plantilla[k].size
            e[k], i = v[i:j], j
        re, im = np.split(v[i:], 2)
        red, i = re + 1j * im, 0
        for k in ("Y", "W", "Z", "Sc"):
            j = i + plantilla[k].size
            e[k], i = red[i:j].reshape(plantilla[k].shape, order="F"), j
        return e

    def cerrar(self):
        for proc in self.procs:
            if proc.poll() is None:
                try:
                    proc.stdin.write("SALIR\n")
                    proc.stdin.flush()
                    proc.wait(timeout=30)
                except Exception:
                    proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cerrar()
