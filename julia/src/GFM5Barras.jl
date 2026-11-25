"Modelo del sistema de 5 barras con inversores GFM (VSG) y BESS."
module GFM5Barras

using LinearAlgebra, TOML
using OrdinaryDiffEqRosenbrock, SciMLBase

include("red.jl")
include("flujo.jl")
include("modelo.jl")
include("simulacion.jl")

const RAIZ = normpath(joinpath(@__DIR__, "..", ".."))
leer_parametros(ruta=joinpath(get(ENV, "GFM_DATOS", joinpath(RAIZ, "datos")), "parametros.toml")) = TOML.parsefile(ruta)

export leer_parametros, construir_sistema, simular, salidas, metricas, evaluar, algebra, rhs!, malla,
       sin_limitador, ilimite, trapz, NG,
       NOMBRES_G, NOMBRES_M, NX, igfm, RAIZ

end
