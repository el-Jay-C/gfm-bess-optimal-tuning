#!/usr/bin/env bash
# Ejecuta todos los experimentos del informe (desde Codigos_final/). Uso: ./ejecutar_todo.sh (o PY=... JOBS=... ./ejecutar_todo.sh)
# Autodetecta Python (rutas relativas) y núcleos del sistema; cada experimento guarda sus CSV y `conv` se reanuda solo.
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
  for cand in ../Codigos/.venv/bin/python .venv/bin/python ../.venv/bin/python ../Codigos/.venv/Scripts/python.exe .venv/Scripts/python.exe; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
  PY="${PY:-$(command -v python3 || command -v python)}"
fi
NPROC="$("$PY" -c "import os; print(os.cpu_count() or 4)")"
JOBS="${JOBS:-$(( NPROC > 2 ? NPROC / 2 : 1 ))}"
[ "$JOBS" -lt 1 ] && JOBS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1   # un hilo por proceso en paralelo: tiempos comparables
mkdir -p resultados/logs
correr() { "$PY" python/ejecutar_experimentos.py $1 > "resultados/logs/$(echo $1 | tr -d ' -' ).log" 2>&1; }
export -f correr; export PY
printf '%s\n' "conv --metodo F4 --n 20 --workers 1" "conv --metodo F3 --n 20 --workers 1" \
  "conv --metodo F5 --n 10 --workers 1" "conv --metodo F1 --n 10 --workers 1" \
  "conv --metodo F2 --n 20 --workers 1" "conv --metodo F4b --n 4 --workers 1" \
  "modos --n 6 --workers 1" "ablacion --n 6 --workers 1" "pesos --n 6 --workers 1" \
  "barridos --n 3 --workers 1" "robusto --n 6 --workers 1" | xargs -P "$JOBS" -I{} bash -c 'correr "{}"'
correr "malla"
correr "optimos"
correr "cert"
julia --threads=auto --project=julia julia/validar.jl > resultados/logs/validar.log 2>&1
date > resultados/logs/TERMINADO
