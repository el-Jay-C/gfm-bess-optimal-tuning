# Instala las librerías de Julia del proyecto en el entorno de esta carpeta.
# Uso (desde Codigos/):  julia julia/instalar_paquetes.jl
using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqTsit5", "SciMLBase"])
Pkg.instantiate()
Pkg.precompile()
println("Paquetes de Julia instalados en ", @__DIR__)
