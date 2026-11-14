"""Optimización por enjambre de partículas (PSO) en la caja [0, 1]^n."""
import numpy as np


def pso(fun, dim, n_part=12, n_iter=1000, c1=2.0, c2=2.0, w0=0.9, w1=0.4, semilla=0, callback=None, z_ini=None):
    """Minimiza fun(z), z ∈ [0,1]^dim. Inercia decreciente lineal w0 → w1 (Shi y Eberhart).
    callback(it, z_mejor, f_mejor) se llama en cada iteración. z_ini (opcional) reemplaza a la primera
    partícula, para que el enjambre parta del mismo punto inicial que los métodos locales."""
    rng = np.random.default_rng(semilla)
    z = rng.random((n_part, dim))
    if z_ini is not None:
        z[0] = np.clip(z_ini, 0.0, 1.0)
    v = np.zeros_like(z)
    fz = np.array([fun(zi) for zi in z])
    pb, fpb = z.copy(), fz.copy()
    gb = pb[np.argmin(fpb)].copy()
    for it in range(n_iter):
        w = w0 - (w0 - w1) * it / max(n_iter - 1, 1)
        r1, r2 = rng.random((2, n_part, dim))
        v = np.clip(w * v + c1 * r1 * (pb - z) + c2 * r2 * (gb - z), -0.2, 0.2)
        z = np.clip(z + v, 0.0, 1.0)
        fz = np.array([fun(zi) for zi in z])
        mejor = fz < fpb
        pb[mejor], fpb[mejor] = z[mejor], fz[mejor]
        gb = pb[np.argmin(fpb)].copy()
        if callback:
            callback(it, gb, fpb.min())
    return gb, fpb.min()


if __name__ == "__main__":      # verificación mínima: esfera desplazada
    z, fz = pso(lambda z: np.sum((z - 0.3) ** 2), 4, n_iter=200)
    assert fz < 1e-6 and np.allclose(z, 0.3, atol=1e-3), (z, fz)
    z, fz = pso(lambda z: np.sum((z - 0.3) ** 2), 4, n_iter=5, z_ini=np.full(4, 0.3))
    assert fz < 1e-12
    print("PSO OK", z, fz)
