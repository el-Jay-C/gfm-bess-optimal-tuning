# Verificaciones del modelo: flujo de potencia, reducción de Kron, equilibrio inicial y limitador.

include(joinpath(@__DIR__, "src", "GFM5Barras.jl"))
using .GFM5Barras, LinearAlgebra, Printf, Random

p = leer_parametros()
s = construir_sistema(p)
println("Flujo de potencia: ", s.it_fp, " iteraciones")
for b in eachindex(s.V)
    @printf("  barra %d: |V| = %.4f pu, θ = %7.3f°, Pg = %7.2f MW, Qg = %7.2f Mvar\n",
            b, abs(s.V[b]), rad2deg(angle(s.V[b])), real(s.Sgen[b]) * s.Sb, imag(s.Sgen[b]) * s.Sb)
end
@printf("  corriente previa: GFM1 %.3f pu, GFM2 %.3f pu\n", s.i0...)
n, o = p["norma"], p["optimizacion"]
@assert all(n["V_min_normal"] .<= abs.(s.V) .<= n["V_max_normal"]) "Tensiones fuera de banda normal"

# x0 es equilibrio para cualquier θ: en x0 se anulan todos los términos que multiplica θ.
# (El SOC no está en equilibrio porque el BESS ya descarga antes del evento; se excluye.)
lb = [o["H_lim"][1], o["H_lim"][1], o["D_lim"][1], o["D_lim"][1], o["Kq_lim"][1], o["Kq_lim"][1], o["K_lim"][1], o["K_lim"][1]]
ub = [o["H_lim"][2], o["H_lim"][2], o["D_lim"][2], o["D_lim"][2], o["Kq_lim"][2], o["Kq_lim"][2], o["K_lim"][2], o["K_lim"][2]]
rng = MersenneTwister(2026)
θs = [Float64.(o["theta_inicial"]), [lb .+ rand(rng, 8) .* (ub .- lb) for _ in 1:50]...]
peor = 0.0
for e in s.esc
    Iiny = -conj.(e.Sc_pre ./ s.V)
    @assert norm(e.Yred_pre * s.E0 - transpose(e.W_pre) * Iiny - s.Ifu) < 1e-9 "Kron no reproduce las corrientes del flujo"
    @assert norm(e.W_pre * s.E0 + e.Z_pre * Iiny - s.V) < 1e-9 "Kron no reproduce las tensiones del flujo"
    for θ in θs
        dx = zeros(NX)
        rhs!(dx, e.x0, (s, merge(e, (Yred=e.Yred_pre, W=e.W_pre, Z=e.Z_pre, Sc=e.Sc_pre)), θ), 0.0)
        dx[[igfm(i) + 5 for i in 1:2]] .= 0
        global peor = max(peor, maximum(abs, dx))
    end
end
@assert peor < 1e-7 "x0 no es equilibrio: max|dx| = $peor"
@printf("Kron y equilibrio inicial para %d sintonías: OK (max|f(x0, θ)| = %.1e)\n", length(θs), peor)

θ = o["theta_inicial"]
for e in s.esc
    t0 = time(); m, g = evaluar(s, e, θ); dt = time() - t0
    @printf("%s (%.3f s):\n", e.nombre, dt)
    println("  ", join([@sprintf("%s=%.4g", k, v) for (k, v) in zip(NOMBRES_M, m)], "  "))
    println("  g: ", join([@sprintf("%s=%.3g", k, v) for (k, v) in zip(NOMBRES_G, g)], "  "))
end

# Limitador: una sintonía que pide más corriente que la nominal agota el presupuesto térmico y luego
# el limitador lleva la corriente a I_cont.
θx = [8.0, 8.0, 50.0, 50.0, 20.0, 20.0, 1.0, 1.0]
t, X = simular(s, s.esc[1], θx; t=collect(0:0.05:120))
y = salidas(s, s.esc[1], t, X)
@printf("Limitador (θ sin restricciones, 120 s): I2 máx = %.3f, I2 final = %.4f, pred2 final = %.3f, ϑ2/B = %.2f\n",
        maximum(r.I2 for r in y), y[end].I2, y[end].pred2, y[end].th2 / s.B)
@assert y[end].I2 < s.Icont + 0.02 "El limitador no llevó la corriente al valor nominal"
