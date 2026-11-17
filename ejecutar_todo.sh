#!/usr/bin/env bash
# Ejecuta el lote v3 desde Codigos_final/, cinco procesos con un hilo cada uno.
set -uo pipefail   # sin -e: un experimento fallido queda en su log y el resto sigue hasta TERMINADO
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
  for cand in ../Codigos/.venv/bin/python .venv/bin/python ../.venv/bin/python ../Codigos/.venv/Scripts/python.exe .venv/Scripts/python.exe; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
  PY="${PY:-$(command -v python3 || command -v python)}"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
# Archiva al arrancar; --backup conserva también archivos homónimos de relanzamientos.
archivo=resultados/v2_2026-11-14
mkdir -p "$archivo/logs" resultados/logs
shopt -s nullglob
csv=(resultados/*.csv)
logs=(resultados/logs/*)
((${#csv[@]} == 0)) || mv --backup=numbered "${csv[@]}" "$archivo/"
((${#logs[@]} == 0)) || mv --backup=numbered "${logs[@]}" "$archivo/logs/"
correr() { "$PY" python/ejecutar_experimentos.py $1 > "resultados/logs/$(echo "$1" | tr -d ' -').log" 2>&1; }
export -f correr
export PY
printf '%s\n' "robusto --n 6 --workers 1" "conv --metodo F4b --n 20 --workers 1" \
  "conv --metodo F3 --n 20 --workers 1" "pareto --n 6 --workers 1" "barridos --n 3 --workers 1" \
  "conv --metodo F4 --n 20 --workers 1" "conv --metodo F5 --n 20 --workers 1" \
  "conv --metodo F1 --n 20 --workers 1" "conv --metodo F2 --n 20 --workers 1" \
  "modos --n 6 --workers 1" "ablacion --n 6 --workers 1" | xargs -P 5 -I{} bash -c 'correr "$1"' _ "{}"
correr "malla"
correr "optimos"
correr "cert"
julia --project=julia julia/validar.jl > resultados/logs/validar.log 2>&1
date > resultados/logs/TERMINADO
