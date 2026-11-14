# Sintonía óptima de inversores Grid-Forming (GFM) con almacenamiento BESS en un sistema de 5 barras

Repositorio de código del proyecto de Optimización Avanzada (**IEE3002**): diseño, formulación multi-método, análisis bajo incertidumbre, certificación de estabilidad por SDP y validación no lineal de controladores para inversores **Grid-Forming (GFM)** respaldados por sistemas de almacenamiento en baterías (**BESS**) en la red de prueba de **5 barras de Stagg y El-Abiad (230 kV)**.

---

## 1. Contexto del problema

La integración masiva de generación fotovoltaica y sistemas basados en inversores desplaza inercia rotatoria síncrona de las redes eléctricas. Este proyecto formula y compara distintas metodologías de optimización para sintonizar simultáneamente los lazos de control primario de frecuencia y tensión de **dos unidades GFM + BESS** (barras 2 y 3) en presencia de un **generador síncrono (SG)** con turbina y gobernador (barra 1) y dos cargas (barras 4 y 5), operando bajo una perturbación de escalón de carga o variación de generación fotovoltaica.

### Sistema modelado (22 estados diferenciales)
- **Red eléctrica:** Sistema de 5 barras de Stagg y El-Abiad (230 kV) con parámetros por kilómetro basados en Kumari et al. (2021) y Bhandakkar & Mathew (2017), reducido algebraicamente mediante **reducción de Kron** a las 3 barras activas (SG + GFM1 + GFM2) para evaluación dinámica exacta y reconstrucción completa de tensiones en las 5 barras (`W`).
- **Generador Síncrono (barra 1, 4 estados):** ángulo $\delta_g$, velocidad $\omega_g$ y dinámica de turbina/gobernador ($x_g, x_r$).
- **Cada unidad GFM + BESS (barras 2 y 3, 9 estados por unidad):** máquina síncrona virtual ($\delta_i, \omega_i$), magnitud de fem interna $E_i$, potencia reactiva filtrada $q_{m,i}$, circuito equivalente **Thevenin de BESS** ($SOC_i$ y rama RC transitoria $v_{rc,i}$), energía acumulada respecto al punto previo $e_i$, integral térmica de corriente $I_{\mathrm{cont}}$ ($\vartheta_i$) y estado de reducción de potencia del **limitador suave de sobrecarga** ($p_{\mathrm{red},i}$).

### Variables de decisión ($\theta \in \mathbb{R}^8$)
$$\theta = [H_1,\; H_2,\; D_1,\; D_2,\; K_{q,1},\; K_{q,2},\; K_1,\; K_2]$$
- $H_1, H_2$: constantes de inercia virtual $[\text{s}]$.
- $D_1, D_2$: coeficientes de amortiguamiento / estatismo activo-frecuencia $[\text{pu}]$.
- $K_{q,1}, K_{q,2}$: ganancias de estatismo reactivo-tensión $Q\text{-}V$.
- $K_1, K_2$: ganancias integrales del lazo de regulación de tensión.

### Restricciones normativas y físicas (`(P)`)
El problema `(P)` incorpora exigencias técnicas inspiradas en la norma chilena (**NTSyCS / Anexo Técnico AT-IBR del CEN**) e **IEEE Std 2800-2022**:
- **R2 (Frecuencia y RoCoF):** desviación máxima $\Delta f_{\max}$ y tasa de cambio de frecuencia máxima permisibles.
- **R3 (Corriente AC e integral térmica):** límite pico instantáneo ($I_{\max} = 1{,}2\text{ pu}$), límite térmico sostenido ($\vartheta_i(T) \le B$) y corriente estacionaria al final del horizonte ($I_i(T) \le I_{\mathrm{cont}}$).
- **R4 (BESS y convertidor):** corriente máxima de batería ($i_{b,i}$), tensión mínima de celda ($v_{\mathrm{bat},i}$), potencia nominal ($p_{b,i} \le 1\text{ pu}$) y rango seguro de $SOC$.
- **R5 (Energía de servicio):** energía extrapolada hasta el relevo del control secundario ($T_s = 300\text{ s}$) acotada por la reserva $E_{\mathrm{res},i}$.
- **R6 (Tensión en barras):** banda de tensión estacionaria y transitoria en las 5 barras.
- **R7, R8a, R8b (Restricciones estáticas sobre inercia):** inercia total mínima del sistema ante contingencia de diseño ($50\text{ MW}$), respaldo energético de reserva inercial y holgura de potencia efectiva por GFM.
- **R9 (Escala temporal Q-V):** acotamiento analítico de $r_i = K_i (K_{q,i}\alpha_i + \beta_i)$ para cumplir tiempo de crecimiento máximo en tensión y separación de escalas frente al filtro $q_m$.

### Modos y formulaciones comparadas
- **Modos de operación:**
  - **Modo R (Restricciones estrictas):** mantiene el limitador inactivo ($\rho_i \equiv 0$) exigiendo límites sostenidos explícitos.
  - **Modo L (Limitador en la dinámica):** habilita el limitador suave de corriente/potencia durante la simulación y elimina las restricciones de punto final que el limitador absorbe, cuantificando el beneficio de la sobrecarga transitoria permitida.
- **Métodos de caja negra (oráculo en Julia con `Rodas5P`):**
  - `F1`: réplica secuencial del artículo base (SQP sobre $(H,D)$ seguido de PSO sobre $(K_q,K)$, sin limitador ni R2–R9).
  - `F2`: SQP conjunto (`SLSQP` con gradientes por diferencias finitas).
  - `F3`: PSO conjunto con penalización cuadrática exterior.
- **Métodos de transcripción directa (CasADi + IPOPT):**
  - `F4`: colocación trapezoidal sobre malla no uniforme de 660 intervalos (14 529 variables, 30 394 restricciones, Hessiano exacto).
  - `F4b`: colocación trapezoidal con aproximación L-BFGS.
  - `F5`: disparo múltiple con 66 intervalos integrados con RK4 (1 461 variables).
- **Optimización bajo incertidumbre (`S1`, `S2`, `S3`):**
  - `minmax` (peor caso), `esperado` ($\mathbb{E}[J]$), `cvar80` ($\mathrm{CVaR}_{0{,}8}$) y `afin_minmax` (regla de decisión afín $\theta(\xi) = \theta^0 + \xi\Delta\theta$ en función de la generación FV previa $\xi$).
- **Certificados SDP (`cert`):**
  - Verificación de razón de amortiguamiento ($\zeta \ge 5\,\%$) y búsqueda vía LMI/SDP (**CVXPY + Clarabel/SCS**) de una **función de Lyapunov común** ($A_s^\top P + PA_s + 2\gamma P \preceq 0$) sobre los equilibrios linealizados de S1–S3.

---

## 2. Portabilidad y paralelismo multiplataforma

Todos los archivos del proyecto utilizan **rutas relativas** detectadas dinámicamente desde la raíz del repositorio (`Path(__file__).resolve().parents[...]` y `@__DIR__`), sin referencias absolutas ni código dependiente del sistema operativo (compatible con **Linux**, **macOS** y **Windows**).

El cálculo paralelo se adapta automáticamente al hardware disponible en cualquier computador:
- **Oráculo en Julia (`OraculoJulia`):** levanta un pool de procesos independientes sincronizados con `queue.Queue` y caché protegido con `threading.Lock`, controlado por `--workers` (`0` = mitad de los hilos de CPU detectados por `os.cpu_count()`) o la variable de entorno `GFM_WORKERS`.
- **Multiarranque en `F2`/`F3` (`ejecutar_experimentos.py`):** evalúa en paralelo los distintos puntos iniciales mediante `ThreadPoolExecutor` alimentando el pool de procesos de Julia sin sobrecarga de hilos por proceso (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).
- **Certificados SDP (`exp_cert`):** resuelve en paralelo los problemas LMI por sintonía mediante `ProcessPoolExecutor` (aislando el estado global de CasADi, seguro tanto bajo `fork` en Linux como `spawn` en Windows).
- **Validación masiva en Julia (`validar.jl`):** usa `Threads.@threads` (activado con `julia --threads=auto`) sobre los escenarios densos, horizonte de 300 s, 500 escenarios Monte Carlo fuera de muestra y escalón de tensión, precalculando las muestras aleatorias de forma determinista.
- **Ejecución por lotes (`ejecutar_todo.sh`):** autodetecta el intérprete de Python y calcula automáticamente `JOBS` a partir de `os.cpu_count()`.

---

## 3. Estructura del repositorio

| Ruta | Descripción |
|---|---|
| `datos/parametros.toml` | Parámetros de la red de 5 barras, SG, convertidores GFM, BESS, límites normativos (NTSyCS/AT-IBR), cotas de $\theta$, malla temporal, pesos de $J$ y definición de escenarios (`S1_carga_b5`, `S2_carga_b4`, `S3_FV_baja`). |
| `datos/red_exportada.toml` | Constantes del punto de operación y matrices reducidas de Kron ($Y_{\text{red}}$, $W$, $G_L$, $x_0$) exportadas automáticamente desde Julia. |
| `julia/Project.toml`, `julia/Manifest.toml` | Entorno reproducible de paquetes Julia. |
| `julia/instalar_paquetes.jl` | Instala `OrdinaryDiffEqRosenbrock`, `OrdinaryDiffEqTsit5` y `SciMLBase` en el entorno local `julia/`. |
| `julia/src/GFM5Barras.jl` | Módulo principal Julia que expone el modelo, flujo de potencia y simulador. |
| `julia/src/red.jl` | Construye $Y_{\text{bus}}$ con modelo $\pi$ de líneas, calcula cargas de impedancia constante y aplica reducción de Kron. |
| `julia/src/flujo.jl` | Solver AC de flujo de potencia por Newton-Raphson en coordenadas polares. |
| `julia/src/modelo.jl` | EDO no lineal de 22 estados (SG + gobernador + VSG + BESS Thevenin + limitador suave) y álgebra de red/convertidores. |
| `julia/src/simulacion.jl` | Simulación stiff con `Rodas5P`, cálculo de métricas ($\Delta f_{\max}$, $V_{\text{dev}}$, pérdidas, alivio de carga) y restricciones normalizadas de `(P)`. |
| `julia/exportar_red.jl` | Resuelve el flujo de potencia base, arma los escenarios y escribe `datos/red_exportada.toml`. |
| `julia/test_modelo.jl` | Test unitario en Julia: flujo AC, conservación en reducción de Kron, equilibrio estacionario y simulación de verificación. |
| `julia/servidor.jl` | Servidor persistente por `stdin`/`stdout` usado por Python para evaluar simulaciones en Julia como oráculo. |
| `julia/validar.jl` | Validación exhaustiva en paralelo (`Threads.@threads`): escenarios S1–S3, malla densa (1 ms), horizonte extendido (300 s), 500 escenarios aleatorios fuera de muestra y escalón de tensión en SG. |
| `python/gfm/datos.py` | Carga de `parametros.toml` y `red_exportada.toml` (invocando `exportar_red.jl` si hace falta), mallas de tiempo y restricciones estáticas R7/R8/R9. |
| `python/gfm/oraculo.py` | Clase `OraculoJulia`: pool multi-worker thread-safe con caché de simulaciones sobre subprocesos `julia/servidor.jl`. |
| `python/gfm/modelo.py` | Réplica simbólica exacta en CasADi del sistema de 22 estados, sensibilidades $\alpha_i, \beta_i$ de R9 e integrador CVODES para inicialización. |
| `python/gfm/pso.py` | Implementación de Particle Swarm Optimization (`pso`) para `F1` y `F3`. |
| `python/gfm/historial.py` | Registrador de trazas de tiempo, iteraciones, evaluaciones, $J$ y violación por corrida. |
| `python/gfm/form_simulacion.py` | Formulación de caja negra `ProblemaCN` y solvers `F1` (`f1_paper`), `F2` (`f2_sqp`) y `F3` (`f3_pso`). |
| `python/gfm/form_directa.py` | Transcripción directa `ProblemaDirecto` en CasADi + IPOPT (`F4`, `F4b`, `F5`), colocación multiescenario y extracción de multiplicadores KKT. |
| `python/gfm/certificado.py` | Linealización automática alrededor del equilibrio post-falla y solución de LMIs de Lyapunov común y región cónica vía CVXPY. |
| `python/test_consistencia.py` | Verifica consistencia numérica entre el modelo en Julia (`Rodas5P`) y la réplica CasADi (`CVODES`). |
| `python/ejecutar_experimentos.py` | Orquestador CLI de todos los experimentos (`conv`, `modos`, `ablacion`, `pesos`, `barridos`, `robusto`, `malla`, `optimos`, `cert`). |
| `python/graficar.py` | Genera las figuras de análisis (convergencia, trayectorias, barridos y nube de Monte Carlo). |
| `python/resumen.py` | Genera tablas LaTeX (`booktabs`) y métricas resumidas a partir de los CSV en `resultados/`. |
| `ejecutar_todo.sh` | Script Bash maestro que ejecuta toda la batería de experimentos, validación en Julia y marca de término. |
| `requirements.txt` | Dependencias de Python (`numpy`, `scipy`, `pandas`, `casadi`, `matplotlib`, `cvxpy`, `clarabel`). |

---

## 4. Requisitos e instalación

### Prerrequisitos
- **Python >= 3.11** (recomendado 3.11–3.13)
- **Julia >= 1.10** (probado con Julia 1.13)

### Instalación paso a paso (desde la raíz del repositorio)

```bash
# 1. Crear y activar entorno virtual de Python
python3 -m venv .venv
source .venv/bin/activate        # En Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Instalar dependencias de Python
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Instalar y precompilar paquetes de Julia en el entorno local julia/
julia julia/instalar_paquetes.jl
```

---

## 5. Cómo ejecutar el proyecto

> **Importante:** Ejecuta todos los comandos ubicado en la carpeta raíz del repositorio.

### A. Pruebas rápidas de verificación
```bash
# 1. Verificar flujo de potencia, reducción de Kron y equilibrio en Julia
julia --project=julia julia/test_modelo.jl

# 2. Exportar matrices de red (se ejecuta automáticamente si cambia parametros.toml)
julia --project=julia julia/exportar_red.jl

# 3. Contrastar el modelo de Julia frente a la réplica simbólica CasADi
.venv/bin/python python/test_consistencia.py
```

### B. Ejecutar experimentos individuales
Sintaxis general:
```bash
.venv/bin/python python/ejecutar_experimentos.py <experimento> [opciones]
```

Opciones principales:
- `--metodo F1|F2|F3|F4|F4b|F5`: formulación para `conv` (por defecto `F2`).
- `--n N`: número de puntos iniciales del hipercubo latino (por defecto `20`).
- `--workers W`: procesos de Julia en paralelo (`0` usa automáticamente la mitad de los hilos de CPU).
- `--rapido`: omite las partes de IPOPT para pruebas rápidas del flujo.
- `--res CARPETA`: carpeta alternativa de salida (por defecto `resultados/`).

Ejemplos:
```bash
# Convergencia de cada formulación en S1 (modo R)
.venv/bin/python python/ejecutar_experimentos.py conv --metodo F2 --n 20
.venv/bin/python python/ejecutar_experimentos.py conv --metodo F4 --n 20

# Comparación Modo R vs. Modo L y ablación de restricciones
.venv/bin/python python/ejecutar_experimentos.py modos --n 6
.venv/bin/python python/ejecutar_experimentos.py ablacion --n 6

# Sensibilidad a pesos y barridos de reserva / tamaño de contingencia
.venv/bin/python python/ejecutar_experimentos.py pesos --n 6
.venv/bin/python python/ejecutar_experimentos.py barridos --n 3

# Optimización robusta y estocástica (S1-S3) + regla afín
.venv/bin/python python/ejecutar_experimentos.py robusto --n 6

# Refinamiento de malla de colocación (factores x1, x2, x4)
.venv/bin/python python/ejecutar_experimentos.py malla

# Consolidar óptimos y calcular certificados SDP/LMI (usa ProcessPoolExecutor)
.venv/bin/python python/ejecutar_experimentos.py optimos
.venv/bin/python python/ejecutar_experimentos.py cert
```

### C. Validación en Julia, pipeline completo, figuras y tablas
```bash
# Validación paralela en Julia (usa todos los hilos disponibles)
julia --threads=auto --project=julia julia/validar.jl

# Ejecutar la batería completa automáticamente (ajusta PY y JOBS si deseas)
./ejecutar_todo.sh
# Ejemplo personalizado: PY=.venv/bin/python JOBS=4 ./ejecutar_todo.sh

# Generar figuras y tablas resumen en carpetas locales de resultados/
.venv/bin/python python/graficar.py resultados/figuras
.venv/bin/python python/resumen.py resultados/tablas
```

---

## 6. Archivos de resultados generados (`resultados/`)

| Archivo | Contenido |
|---|---|
| `conv_<M>.csv`, `corridas_<M>.csv` | Trazas iteración a iteración y resumen por corrida (tiempo, evaluaciones, $J$ y `viol_real`) para `M` $\in$ `F1..F5`. La ejecución de `conv` se reanuda automáticamente si ya existen corridas previas. |
| `modos.csv` | Comparación de óptimos, restricciones activas y multiplicadores KKT entre Modo R y Modo L (`F2` y `F4`). |
| `ablacion.csv` | Impacto incremental de activar cada grupo de restricciones (`R2` a `R9`) sobre la formulación base `F1`. |
| `pesos.csv` | Sensibilidad frente a distintos pesos $(w_f, w_v, w_\ell)$, ganancia $K_q$ fija y efecto de dependencia de tensión en cargas. |
| `barridos.csv` | Barridos de reserva energética $\rho \in [0{,}03; 0{,}15]$ (incluyendo contraste de $\mathrm{d}J^*/\mathrm{d}\rho$ por multiplicadores KKT vs. diferencias centradas) y tamaño de contingencia $\Delta P \in [20, 60]\text{ MW}$. |
| `robusto.csv` | Óptimos individuales por escenario y soluciones multi-escenario (`minmax`, `esperado`, `cvar80`, `afin_minmax`, `minmax_F4`). |
| `malla.csv` | Refinamiento de malla temporal (`factor` $\times 1, \times 2, \times 4$) para `F4`. |
| `optimos.csv` | Tabla consolidada de las sintonías finales comparadas. |
| `certificados.csv` | Razón de amortiguamiento mínima ($\zeta_{\min}$), tasa de decaimiento por autovalores ($\alpha_{\text{eig}}$) y tasas certificadas por SDP/Lyapunov común ($\alpha_{\text{LMI}}$, $\zeta_{\text{LMI}}$). |
| `validacion.csv`, `validacion_densa.csv` | Validación de cada sintonía en S1–S3 sobre malla estándar y malla densa de 1 ms. |
| `largo_plazo.csv`, `escalon_tension.csv`, `fuera_muestra.csv` | Respuesta a 300 s, verificación ante escalón de tensión en SG y evaluación Monte Carlo en 500 escenarios aleatorios fuera de muestra. |
| `trayectorias.csv` | Series temporales completas generadas por `validar.jl` (omitidas de Git por su tamaño). |
