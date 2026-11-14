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
- **Script maestro (`ejecutar_todo.sh`):** autodetecta el entorno Python relativo y escala `JOBS` automáticamente según los núcleos disponibles.

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

### B. Pipeline completo automático (Linux / macOS)
```bash
./ejecutar_todo.sh
# Puedes sobrescribir el intérprete o nivel de concurrencia si lo deseas:
# PY=.venv/bin/python JOBS=4 ./ejecutar_todo.sh
```
El script autodetecta los núcleos disponibles, ejecuta en paralelo los experimentos independientes con `--workers 1` por proceso, corre luego `malla`, `optimos`, `cert` y `julia --threads=auto --project=julia julia/validar.jl`, y escribe `resultados/logs/TERMINADO` al terminar. Los experimentos `conv` y `malla` reanudan corridas ya guardadas; para repetir desde cero, respalda o limpia `resultados/`.

### C. Ejecución comando por comando (Windows PowerShell o corridas individuales)
Al ejecutar un experimento individualmente sin paralelismo externo (`xargs`), omitir `--workers` (o usar `--workers 0`) activa automáticamente el pool paralelo según los núcleos de tu equipo:

```powershell
$pyGfm = '.\.venv\Scripts\python.exe'
& $pyGfm python/ejecutar_experimentos.py conv --metodo F1 --n 10
& $pyGfm python/ejecutar_experimentos.py conv --metodo F2 --n 20
& $pyGfm python/ejecutar_experimentos.py conv --metodo F3 --n 20
& $pyGfm python/ejecutar_experimentos.py conv --metodo F4 --n 20
& $pyGfm python/ejecutar_experimentos.py conv --metodo F4b --n 4
& $pyGfm python/ejecutar_experimentos.py conv --metodo F5 --n 10
& $pyGfm python/ejecutar_experimentos.py modos --n 6
& $pyGfm python/ejecutar_experimentos.py ablacion --n 6
& $pyGfm python/ejecutar_experimentos.py pesos --n 6
& $pyGfm python/ejecutar_experimentos.py barridos --n 3
& $pyGfm python/ejecutar_experimentos.py robusto --n 6
& $pyGfm python/ejecutar_experimentos.py malla
& $pyGfm python/ejecutar_experimentos.py optimos
& $pyGfm python/ejecutar_experimentos.py cert
julia --threads=auto --project=julia julia/validar.jl
```

### D. Figuras y tablas
```bash
# Especifica una carpeta local de salida (ej. resultados/figuras y resultados/tablas):
.venv/bin/python python/graficar.py resultados/figuras
.venv/bin/python python/resumen.py resultados/tablas
```
(Si no se pasa argumento, `graficar.py` y `resumen.py` escriben en `../Informe/figuras` y `../Informe/tablas` respectivamente).

---

## 6. CSV producidos y tiempos orientativos

| Comando | CSV generado en `resultados/` | Tiempo orientativo |
|---|---|---|
| `conv --metodo M` | `corridas_M.csv`, `conv_M.csv` | Medianas por corrida: F1 29,4 s; F2 10,0 s; F3 162,6 s; F4 79,4 s; F4b 507,5 s; F5 121,1 s. |
| `modos` | `modos.csv` | Minutos; compara Modo R vs. Modo L (SQP/colocación y multiplicadores KKT). |
| `ablacion` | `ablacion.csv` | Decenas de minutos; agrega grupos de restricciones (`R2` a `R9`) paso a paso. |
| `pesos` | `pesos.csv` | Decenas de minutos; sensibilidad a $(w_v, w_l)$ y casos con $K_q$ fijo. |
| `barridos` | `barridos.csv` | Decenas de minutos; barrido de reserva $\rho$ (multiplicadores vs. diferencias) y tamaño de escalón R/L. |
| `robusto` | `robusto.csv` | Decenas de minutos; óptimos por escenario, `minmax`, `esperado`, `cvar80`, `afin_minmax` y `minmax_F4`. |
| `malla` | `malla.csv` | Minutos; factores $\times 1, \times 2, \times 4$ (resolución 29,8 / 161,4 / 300,9 s más construcción). |
| `optimos` | `optimos.csv` | Segundos; consolida las 8 sintonías finales comparadas. |
| `cert` | `certificados.csv` | Segundos–minutos; amortiguamiento por autovalores y LMI de Lyapunov común. |
| `julia/validar.jl` | `validacion.csv`, `validacion_densa.csv`, `largo_plazo.csv`, `escalon_tension.csv`, `fuera_muestra.csv`, `trayectorias.csv` | Minutos, según núcleos disponibles (`--threads=auto`). |

Los tiempos provienen de `corridas_F*.csv` y `malla.csv` medidos con procesos simultáneos en un procesador de 6 núcleos (12 hilos); el lote completo toma aproximadamente 2–4 horas en paralelo según la máquina.

---

## 7. Lectura correcta de los resultados conservados

- `optimos.csv` consolida el óptimo R de colocación (`F4`, $J = 0{,}63357$ sin limitador), no confundirlo con el óptimo R de SQP en `modos.csv` ($J = 0{,}63164$).
- La tabla generada para sintonías combina métricas **con** limitador activo y la violación máxima de trayectoria **sin** limitador; la tabla de ablación reporta la violación de cada etapa y `viol_P` del problema completo.
- `validacion_densa.csv` y el campo `factible` de `fuera_muestra.csv` verifican trayectorias sin limitador y no incluyen cotas de ganancias ni `R7`–`R9`; por ello no certifican factibilidad completa en Modo L ni para la regla afín.
- La etapa `R7`–`R8` de `ablacion.csv` y los criterios de sintonía única (`minmax`, `esperado`, `cvar80`, `minmax_F4` que alcanzó límite de CPU) resultaron infactibles; no interpretar sus costos como óptimos admisibles.
- La regla afín (`afin_minmax`) es factible en la malla de diseño, pero presenta una violación de $0{,}2523\,\%$ en malla densa de 1 ms en S2, amortiguamiento mínimo de $4{,}82\,\%$ (< $5\,\%$) y tasa común `-inf` (con advertencia numérica en `certificados.csv`); esto indica ausencia de certificado común aceptado, no una demostración de inestabilidad.
- A $T_s = 300\text{ s}$ no se observa un segundo descenso apreciable de frecuencia ni agotamiento del presupuesto térmico en los escenarios evaluados.
