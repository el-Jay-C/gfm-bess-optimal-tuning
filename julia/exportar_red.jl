# Exporta el punto de operación y la red reducida (por escenario) para el modelo de optimización en Python.
# Uso (desde Codigos_final/): julia --project=julia julia/exportar_red.jl
include(joinpath(@__DIR__, "src", "GFM5Barras.jl"))
using .GFM5Barras, TOML

s = construir_sistema(leer_parametros())
mat(A) = [collect(r) for r in eachrow(A)]
cte = Dict(string(k) => getfield(s, k) for k in
           (:f0, :wb, :Sb, :Sg, :Hg, :Dg, :Rg, :TG, :TR, :FH, :Eg, :pm0, :S, :Xf, :tau_q, :Rconv,
            :p0, :q0, :U0, :i0, :Pn, :Eh, :ocv0, :ocv1, :R0, :R1, :tau1, :eta, :Eres, :Imax, :Icont, :B,
            :klim, :tau_lib, :eps_i, :eps_th, :mth, :ath, :ploss0))
cte["Vpre"] = abs.(s.V)
esc = [Dict("nombre" => e.nombre, "dP" => e.dP, "x0" => e.x0, "GL" => e.GL, "pfv" => e.pfv, "pb0" => e.pb0,
            "Yred_re" => mat(real(e.Yred)), "Yred_im" => mat(imag(e.Yred)),
            "W_re" => mat(real(e.W)), "W_im" => mat(imag(e.W))) for e in s.esc]
ruta = joinpath(RAIZ, "datos", "red_exportada.toml")
open(ruta, "w") do io
    TOML.print(io, Dict("constantes" => cte, "escenario" => esc))
end
println("Escrito ", ruta)
