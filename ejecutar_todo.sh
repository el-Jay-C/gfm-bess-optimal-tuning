#!/usr/bin/env bash
# Ejecuta el lote v4 desde Codigos_final/, hasta siete procesos con un hilo cada uno.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
if [ -z "${PY:-}" ]; then
  for cand in ../Codigos/.venv/bin/python .venv/bin/python ../.venv/bin/python \
              ../Codigos/.venv/Scripts/python.exe .venv/Scripts/python.exe; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
  PY="${PY:-$(command -v python3 || command -v python)}"
fi
export PY OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
logs=resultados/logs
export logs
mkdir -p "$logs" || exit 1
rm -f "$logs/TERMINADO" "$logs/FALLIDOS" "$logs/robusto.fin" || exit 1

ruta_log() {
  case "$1" in
    rango|sens|conv) printf '%s/v4_%s_%s.log\n' "$logs" "$1" "$3" ;;
    *) printf '%s/v4_%s.log\n' "$logs" "$1" ;;
  esac
}

termino() {
  [[ -f "$2" && "$(tail -n 1 "$2")" == *"Listo: $1" ]]
}

correr() {
  local exp ruta
  exp="$1"
  ruta=$(ruta_log "$@")
  termino "$exp" "$ruta" && { [[ "$exp" == robusto ]] && touch "$logs/robusto.fin"; return 0; }
  # sens compara con la sintonía base: espera a robusto (va al final de la cola, no bloquea a otros)
  if [[ "$exp" == sens ]]; then until [[ -f "$logs/robusto.fin" ]]; do sleep 20; done; fi
  if [[ "$exp" == validar ]]; then
    julia --project=julia julia/validar.jl > "$ruta" 2>&1
  else
    "$PY" python/ejecutar_experimentos.py "$@" --workers 1 > "$ruta" 2>&1
  fi
  if [[ "$exp" == robusto ]]; then touch "$logs/robusto.fin"; fi
}

export -f ruta_log termino correr
fase1=("rango --problema L_todos --n 6" "rango --problema L_S1 --n 6" "rango --problema R_S3 --n 6")
# xargs reparte en este orden: primero la comparación de convergencia (F4, F2, F1)
fase2=("conv --metodo F4 --n 20" "conv --metodo F2 --n 20" "conv --metodo F1 --n 20" "robusto --n 3"
       "barridos --n 3" "ablacion --n 4" "modos --n 6" "pareto --n 4" "fase1 --n 6")
for caso in klim1 klim4 kp0 kp1 fracP0 fracP1 Kqmin5 nsep3 nsep8; do
  fase2+=("sens --caso $caso --n 2")
done
fase3=(malla optimos cert validar)

paralelo() {
  printf '%s\n' "$@" | xargs -r -P 7 -I{} bash -c 'read -ra paso <<< "$1"; correr "${paso[@]}"' _ "{}"
}

registrar_fallos() {
  local tarea paso ruta
  for tarea in "$@"; do
    read -ra paso <<< "$tarea"
    ruta=$(ruta_log "${paso[@]}")
    termino "${paso[0]}" "$ruta" || printf '%s\n' "$tarea"
  done
}

paralelo "${fase1[@]}"
if ! correr kappa || ! termino kappa "$(ruta_log kappa)"; then
  registrar_fallos "${fase1[@]}" kappa > "$logs/FALLIDOS"
  exit 1
fi
paralelo "${fase2[@]}"
for exp in "${fase3[@]}"; do
  correr "$exp"
done

fallos=$(registrar_fallos "${fase1[@]}" kappa "${fase2[@]}" "${fase3[@]}")
if [[ -n "$fallos" ]]; then
  printf '%s\n' "$fallos" > "$logs/FALLIDOS"
  exit 1
fi
date > "$logs/TERMINADO"
