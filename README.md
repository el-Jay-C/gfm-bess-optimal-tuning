# Sintonía óptima de inversores Grid-Forming (GFM) con almacenamiento BESS en un sistema de 5 barras

Repositorio de código del proyecto de Optimización Avanzada (**IEE3002**): modelamiento dinámico en Julia (`Rodas5P`), optimización multi-método en Python (SciPy SLSQP, PSO, CasADi + IPOPT y CVXPY + Clarabel/SCS) y validación no lineal para sintonizar controladores de inversores **Grid-Forming (GFM)** con sistemas de almacenamiento en baterías (**BESS**) en la red de **5 barras de Stagg y El-Abiad (230 kV, 50 Hz)**.

> Los CSV incluidos en `resultados/` corresponden a los resultados definitivos conservados. Generar figuras y tablas no requiere volver a correr los experimentos. Los estados `Solve_Succeeded` y el archivo `TERMINADO` no implican factibilidad por sí solos: revisar siempre `viol`, `viol_real` y las notas de la Sección 7.

---

## 1. Contexto del problema

La sustitución de generación síncrona por recursos basados en inversores reduce la inercia rotatoria y exige diseñar sintonías que cumplan criterios de estabilidad, capacidad térmica/corriente y reserva energética. Este proyecto modela y compara formulaciones de optimización dinámica para ajustar simultáneamente los lazos de control primario de frecuencia y tensión de **dos inversores GFM con BESS** (barras 2 y 3) en presencia de un **generador síncrono (SG)** térmico con turbina y gobernador (barra 1) y cargas de impedancia constante (barras 4 y 5).

### Sistema modelado (22 estados diferenciales)
- **Red eléctrica:** Sistema de 5 barras (230 kV, 50 Hz) con parámetros de línea de Kumari et al. (2021), reducido algebraicamente por **reducción de Kron** a los 3 nodos internos activos (`Yred`, reconstrucción completa a las 5 barras mediante `W`).
- **Generador síncrono (barra 1, 4 estados):** $\delta_g, \omega_g$ y turbina/gobernador con recalentador ($x_g, x_r$).
- **Cada unidad GFM + BESS (barras 2 y 3, 9 estados por unidad):** máquina síncrona virtual ($\delta_i, \omega_i$), fem interna $E_i$, potencia reactiva filtrada $q_{m,i}$, circuito equivalente **Thevenin de BESS** ($SOC_i$ y rama RC transitoria $v_{rc,i}$), energía acumulada respecto al punto previo $e_i$, integral térmica $I^2t$ ($\vartheta_i$) y estado de reducción de consigna del **limitador suave de sobrecarga** ($p_{\mathrm{red},i}$).

### Variables de decisión ($\theta \in \mathbb{R}^8$)
$$\theta = [H_1,\; H_2,\; D_1,\; D_2,\; K_{q,1},\; K_{q,2},\; K_1,\; K_2]$$
- $H_1, H_2$: constantes de inercia virtual $[\text{s}]$.
- $D_1, D_2$: estatismos / amortiguamientos activos $[\text{pu}]$.
- $K_{q,1}, K_{q,2}$: estatismos del lazo reactivo $Q\text{-}V$.
- $K_1, K_2$: ganancias integrales de regulación de tensión $[\text{s}^{-1}]$.

### Restricciones normativas y físicas `(P)`
- **R2 (Frecuencia y RoCoF):** desviación máxima $\Delta f_{\max} \le 1\text{ Hz}$ y $\mathrm{RoCoF} \le 2\text{ Hz/s}$ en ventana de 500 ms (NTSyCS 3-10).
- **R3 (Corriente AC y presupuesto térmico):** pico instantáneo $i_i(t) \le I_{\max} = 1{,}3\text{ pu}$ (AT-IBR 3-1 d), integral térmica $\vartheta_i(T) \le B = 3{,}45\text{ pu}^2\text{s}$ y corriente final $i_i(T) \le I_c = 1\text{ pu}$.
- **R4 (BESS y convertidor):** límites de corriente de batería ($|i_{b,i}| \le 1{,}2$), tensión mínima ($v_{b,i} \ge 0{,}85$), potencia nominal ($|p_{b,i}| \le 1\text{ pu}$) y rango de $SOC_i \in [0{,}1; 0{,}9]$.
- **R5 (Energía de servicio):** extrapolación hasta el relevo secundario $T_s = 300\text{ s}$ acotada por la reserva del 5 % ($E_{\mathrm{res}}$).
- **R6 (Tensión en barras):** banda $0{,}93\text{–}1{,}07\text{ pu}$ (NTSyCS 5-23).
- **R7, R8a, R8b (Restricciones estáticas de inercia):** inercia mínima del sistema ante contingencia de diseño ($\Delta P = 50\text{ MW}$), respaldo de reserva energética e inercia efectiva dentro de la holgura de potencia (AT-IBR 7-6 k).
- **R9–R10 (Escala temporal Q–V y cotas):** rapidez mínima $r_i = K_i(K_{q,i}\alpha_i + \beta_i) \ge 2{,}2$ (tiempo de crecimiento $< 1\text{ s}$, AT-IBR 3-3 e) y separación de escalas $\tau_v \ge 5\tau_q$, más cotas de ganancias.

### Modos, formulaciones y certificación
- **Modo R vs. Modo L:** en **Modo R** se exige $|p_{b,i}|\le 1$, $\vartheta_i(T)\le B$ e $i_i(T)\le I_c$ sin activar el limitador; en **Modo L** se incorporan las ecuaciones suaves del limitador en la dinámica permitiendo sobrecarga transitoria acotada por corriente.
- **Métodos secuenciales (caja negra con oráculo Julia `Rodas5P`):** `F1` (réplica secuencial del artículo base: SQP + PSO sin limitador ni R2–R9), `F2` (SQP conjunto `SLSQP`), `F3` (PSO penalizado).
- **Métodos simultáneos (CasADi + IPOPT):** `F4` (colocación trapezoidal, 14 529 variables, Hessiano exacto), `F4b` (colocación con L-BFGS), `F5` (disparo múltiple con 66 intervalos RK4, 1 461 variables).
- **Incertidumbre sobre escenarios `S1`, `S2`, `S3`:** sintonía única bajo peor caso (`minmax`), valor esperado (`esperado`), `CVaR` ($\alpha = 0{,}8$, `cvar80`) y regla de decisión afín con recurso (`afin_minmax`, $\theta(\xi) = \theta^0 + \xi\Delta\theta$ según la generación FV disponible previa al evento).
- **Certificación de pequeña señal (`cert`):** amortiguamiento modal ($\zeta \ge 5\,\%$) y búsqueda vía SDP/LMI (**CVXPY + Clarabel/SCS**) de una función cuadrática de Lyapunov común sobre los equilibrios AC linealizados.

---

## 2. Portabilidad y paralelismo automático

El proyecto es **100 % portable** (Linux, macOS y Windows) sin rutas absolutas ni suposiciones de hardware:
- Todas las rutas se resuelven de forma relativa desde la raíz del código (`Path(__file__).resolve().parents[...]` en Python y `@__DIR__` en Julia).
- **`--workers` (por defecto `0` / `GFM_WORKERS`):** autodetecta la mitad de los hilos de CPU vía `os.cpu_count()` para levantar un pool thread-safe de procesos `julia/servidor.jl` (`OraculoJulia`) consumido en paralelo por `ThreadPoolExecutor` en multiarranques de `F2`/`F3`.
- **Certificados SDP (`cert`):** usa `ProcessPoolExecutor` hasta `os.cpu_count()` procesos para aislar el estado global de CasADi de forma segura bajo `fork` (Linux/macOS) y `spawn` (Windows).
- **Validación masiva (`julia/validar.jl`):** paraleliza con `Threads.@threads` (activado con `julia --threads=auto`) manteniendo determinismo en las muestras de Monte Carlo.
- **Script maestro (`ejecutar_todo.sh`):** autodetecta el entorno Python relativo y corre hasta siete experimentos a la vez (`xargs -P 7`), cada uno con un hilo.

---

## 3. Carpetas y scripts

| Ruta | Función |
|---|---|
| `datos/parametros.toml` | Fuente única de parámetros físicos, límites normativos, escenarios, pesos, cotas y malla temporal. |
| `datos/red_exportada.toml` | Punto de operación y matrices reducidas por Kron exportadas por Julia. |
| `julia/Project.toml`, `Manifest.toml` | Entorno reproducible de paquetes Julia. |
| `julia/instalar_paquetes.jl` | Instala y precompila `OrdinaryDiffEqRosenbrock`, `OrdinaryDiffEqTsit5` y `SciMLBase`. |
| `julia/exportar_red.jl` | Resuelve flujo AC por escenario y exporta `datos/red_exportada.toml`. |
| `julia/src/GFM5Barras.jl` | Módulo principal del simulador en Julia. |
| `julia/src/flujo.jl`, `red.jl` | Flujo de potencia Newton-Raphson AC, modelo $\pi$ de líneas y reducción de Kron. |
| `julia/src/modelo.jl` | EDO de 22 estados: SG, VSG, BESS Thevenin y limitador suave de sobrecarga. |
| `julia/src/simulacion.jl` | Integración stiff `Rodas5P`, cálculo de métricas y restricciones normalizadas de `(P)`. |
| `julia/servidor.jl` | Oráculo persistente por `stdin`/`stdout` para evaluaciones desde Python. |
| `julia/test_modelo.jl` | Prueba unitaria de flujo AC, reducción de Kron, equilibrio y limitador. |
| `julia/validar.jl` | Validación en paralelo (`Threads.@threads`): escenarios, malla de 1 ms, horizonte de 300 s, 500 muestras Monte Carlo y escalón de tensión. |
| `python/gfm/datos.py` | Carga `parametros.toml`/`red_exportada.toml` (invocando `exportar_red.jl` si cambia el TOML), mallas y R7–R9. |
| `python/gfm/modelo.py` | Réplica simbólica exacta en CasADi, sensibilidades Q–V ($\alpha_i, \beta_i$) e integrador CVODES. |
| `python/gfm/oraculo.py` | Pool multi-worker thread-safe (`OraculoJulia`) con caché compartida y sincronización de escenarios. |
| `python/gfm/form_simulacion.py` | Formulación `ProblemaCN` y métodos `F1`, `F2` SQP (`SLSQP`), `F3` PSO y criterios multiescenario. |
| `python/gfm/form_directa.py` | Transcripción directa `ProblemaDirecto` en CasADi + IPOPT (`F4`, `F4b`, `F5`) y multiplicadores KKT. |
| `python/gfm/pso.py`, `historial.py` | Algoritmo PSO y registro de convergencia por corrida. |
| `python/gfm/certificado.py` | Linealización en equilibrio AC y LMIs de amortiguamiento/Lyapunov común (CVXPY + Clarabel/SCS). |
| `python/test_consistencia.py` | Contrasta trayectorias entre Julia (`Rodas5P`) y CasADi (`CVODES`). |
| `python/ejecutar_experimentos.py` | CLI principal (`conv`, `modos`, `ablacion`, `pesos`, `barridos`, `robusto`, `malla`, `optimos`, `cert`). |
| `python/graficar.py` | Genera `trayectorias_final.pdf`, `convergencia_final.pdf` y `barridos_final.pdf`. |
| `python/resumen.py` | Genera las tablas LaTeX (`sintonias.tex`, `ablacion.tex`, `convergencia.tex`, `robusto.tex`) e imprime el resumen en stdout. |
| `ejecutar_todo.sh` | Pipeline Bash completo de experimentos, certificados, validación paralela y marca `TERMINADO`. |
| `requirements.txt` | Dependencias Python (`numpy`, `scipy`, `pandas`, `casadi`, `matplotlib`, `cvxpy`, `clarabel`). |

---

## 4. Instalación paso a paso

Requiere **Python $\ge$ 3.11**, **Julia $\ge$ 1.10** (probado en 1.13) y, si además se desea compilar un informe LaTeX, TeX Live (`pdflatex` + `bibtex`). Ejecutar desde la raíz de este repositorio:

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
julia --project=julia julia/instalar_paquetes.jl
julia --project=julia julia/exportar_red.jl
```

### Windows (PowerShell)
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
julia --project=julia julia/instalar_paquetes.jl
julia --project=julia julia/exportar_red.jl
```

> Conserva `julia/Manifest.toml` para reproducir versiones exactas. La primera instalación y precompilación de paquetes Julia requiere conexión a Internet.

---

## 5. Cómo ejecutar el proyecto

### A. Pruebas rápidas de verificación
```bash
julia --project=julia julia/test_modelo.jl
.venv/bin/python python/test_consistencia.py      # En Windows: .\.venv\Scripts\python.exe python/test_consistencia.py
```

### B. Lote completo v4 (Linux / macOS)
```bash
./ejecutar_todo.sh
# Para fijar el intérprete:
# PY=.venv/bin/python ./ejecutar_todo.sh
```
El script lanza hasta **siete procesos simultáneos** (`xargs -P 7`), cada uno con `--workers 1` y un hilo de
BLAS y de Julia, en tres fases:

1. `rango` (`L_todos`, `L_S1`, `R_S3`) y luego `kappa` (elección de $\kappa^\star$).
2. `conv` con F4, F2 y F1 (20 arranques cada uno, F2 y F4 con parada temprana de paciencia 5), `robusto`,
   `barridos`, `ablacion`, `modos`, `pareto`, `fase1` y las nueve sensibilidades `sens --caso ...`
   (`sens` espera a que termine `robusto`).
3. `malla`, `optimos`, `cert` y `julia/validar.jl`.

Al terminar escribe `resultados/logs/TERMINADO`, o `resultados/logs/FALLIDOS` con las tareas que no
llegaron a `Listo`. Cada experimento escribe su log en `resultados/logs/v4_*.log`. `conv` y `malla` reanudan
las corridas ya guardadas; para repetir desde cero, respalda o limpia `resultados/`.

En el computador del proyecto (6 núcleos, 12 hilos) el lote completo tomó unas **7 h**.

### C. F4 sin parada temprana (fila de control de la tabla de convergencia)
No forma parte de `ejecutar_todo.sh`. Se reparte en siete procesos con `--parte i/7` y luego se juntan las
partes:
```bash
for i in 0 1 2 3 4 5 6; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
  .venv/bin/python python/ejecutar_experimentos.py conv --metodo F4 --n 20 --paciencia 0 --parte $i/7 \
    --workers 1 > resultados/logs/v4_conv_F4_sin_paciencia_parte$i.log 2>&1 &
done
wait
mkdir -p resultados/partes_F4_sin_paciencia
mv resultados/*_F4_sin_paciencia_parte*.csv resultados/partes_F4_sin_paciencia/
.venv/bin/python - <<'PY'
import glob, pandas as pd
for t in ("corridas", "conv"):
    partes = sorted(glob.glob(f"resultados/partes_F4_sin_paciencia/{t}_F4_sin_paciencia_parte*.csv"))
    pd.concat(map(pd.read_csv, partes)).sort_values("corrida").to_csv(f"resultados/{t}_F4_sin_paciencia.csv", index=False)
PY
```
Tomó unos 22 min. `--paciencia 0` desactiva la parada temprana (sirve también con F2); `--parte i/m` corre
sólo las corridas `k` con `k % m == i` y agrega el sufijo `_parte{i}` a los archivos.

### D. Ejecución comando por comando
Sin paralelismo externo, omitir `--workers` (o `--workers 0`) usa la mitad de los hilos de la CPU:

```bash
PY=.venv/bin/python        # En Windows: $PY = '.\.venv\Scripts\python.exe'
$PY python/ejecutar_experimentos.py rango --problema L_todos --n 6   # también L_S1 y R_S3
$PY python/ejecutar_experimentos.py kappa
$PY python/ejecutar_experimentos.py conv --metodo F1 --n 20          # también F2 y F4
$PY python/ejecutar_experimentos.py robusto --n 3
$PY python/ejecutar_experimentos.py barridos --n 3
$PY python/ejecutar_experimentos.py ablacion --n 4
$PY python/ejecutar_experimentos.py modos --n 6
$PY python/ejecutar_experimentos.py pareto --n 4
$PY python/ejecutar_experimentos.py fase1 --n 6
$PY python/ejecutar_experimentos.py sens --caso kp0 --n 2           # klim1 klim4 kp0 kp1 fracP0 fracP1 Kqmin5 nsep3 nsep8
$PY python/ejecutar_experimentos.py malla
$PY python/ejecutar_experimentos.py optimos
$PY python/ejecutar_experimentos.py cert
julia --threads=auto --project=julia julia/validar.jl
```

### E. Figuras y tablas
```bash
.venv/bin/python python/graficar.py resultados/figuras
.venv/bin/python python/resumen.py resultados/tablas
```
Sin argumento, `graficar.py` y `resumen.py` escriben en `../Informe/figuras` y `../Informe/tablas`.

---

## 6. De qué CSV sale cada tabla y figura del informe final

Todos los parámetros están en `datos/parametros.toml`.

| Informe final | CSV en `resultados/` |
|---|---|
| Tabla IV (ablación) | `ablacion.csv` |
| Tabla V (sintonías finales) | `optimos.csv`, `validacion.csv`, `certificados.csv` |
| Tabla VI y Fig. 3 (convergencia) | `corridas_F1.csv`, `corridas_F2.csv`, `corridas_F4.csv`, `corridas_F4_sin_paciencia.csv` y sus `conv_*.csv` |
| Tabla VII (una sintonía para S1–S4) | `robusto.csv`, `validacion.csv`, `fuera_muestra.csv` |
| Tabla VIII (sensibilidad) | `sens_*.csv` |
| Fig. 2 (trayectorias) | `trayectorias.csv` (de `validar.jl`) |
| Fig. 4 (barridos) | `barridos.csv` |
| Validación de la solución | `malla.csv`, `validacion_densa.csv`, `largo_plazo.csv`, `escalon_tension.csv`, `certificados.csv` |

Tiempos medianos por corrida en la comparación de convergencia: F1 415 s, F2 10 s, F4 127 s (con parada
temprana) y 371 s (sin ella), medidos con siete procesos simultáneos; sirven para comparar métodos, no como
valores absolutos.

---

## 7. Lectura correcta de los resultados

- Una sintonía es factible si su violación relativa, re-evaluada en Julia con todas las restricciones, no
  supera `1e-3`. Leer `viol_real` con precisión completa: valores como `5.6e-11` son factibles.
- `resultados/descartados_F4_ultimo_iterado/` guarda las corridas de F4 anteriores a la corrección de
  `form_directa.py` (devolvían el último iterado en vez del mejor punto guardado); `descartados_sin_paciencia/`
  guarda una comparación antigua. Ninguna de las dos se usa en el informe.
- En `corridas_F4_sin_paciencia.csv`, las corridas 6 y 8 convergen pero superan el tope de 600 s tras un
  solo corte de R12 y quedan con $\zeta = 6{,}8\,\%$; por eso son infactibles.
- Los criterios `minmax`, `esperado` y `cvar80` de `robusto.csv` dan la misma sintonía: el conjunto factible
  se reduce a un vértice.
