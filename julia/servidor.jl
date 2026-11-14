# Oráculo de simulación para los optimizadores de Python (protocolo de texto por stdin/stdout).
#   -> "EVAL <escenario> <lim> θ1 ... θ8"            <- "<métricas...> <g...>"   (lim = 1 con limitador, 0 sin él)
#   -> "NUEVO <barra> <dP_MW> <SOC1> <SOC2> <Pfv2>"  <- índice del escenario creado
#   -> "SALIR"
include(joinpath(@__DIR__, "src", "GFM5Barras.jl"))
using .GFM5Barras

s = construir_sistema(leer_parametros())
escs = collect(Any, s.esc)
evaluar(s, escs[1], leer_parametros()["optimizacion"]["theta_inicial"])   # compila antes de atender
println("LISTO ", join(NOMBRES_M, " "), " | ", join(NOMBRES_G, " "))
flush(stdout)

for linea in eachline(stdin)
    tok = split(linea)
    isempty(tok) && continue
    tok[1] == "SALIR" && break
    if tok[1] == "NUEVO"
        v = parse.(Float64, tok[2:end])
        push!(escs, s.nuevo_esc(Dict("nombre" => "esc$(length(escs) + 1)", "barra" => Int(v[1]), "dP" => v[2],
                                     "SOC0" => v[3:4], "P_fv2" => v[5])))
        println(length(escs))
    else
        e = merge(escs[parse(Int, tok[2])], (lim=parse(Float64, tok[3]),))
        m, g = evaluar(s, e, parse.(Float64, tok[4:end]))
        println(join(string.([m; g]), " "))
    end
    flush(stdout)
end
