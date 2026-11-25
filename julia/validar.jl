# Valida con el modelo no lineal de Julia las sintonías de resultados/optimos.csv
# (columnas: sintonia, escenario, H1..K2 y la pendiente dH1..dK2 de la regla afín, cero si no aplica).
# Uso (desde Codigos_final/): julia --threads=auto --project=julia julia/validar.jl
# Salidas en resultados/:
#   validacion.csv        cada sintonía en S1-S3, con el limitador (lim=1) y sin él (lim=0)
#   validacion_densa.csv  violación máxima en la malla de optimización frente a una malla de 1 ms
#   largo_plazo.csv       300 s con limitador: régimen permanente, energía real frente a la extrapolada
#   trayectorias.csv      S1, 60 s, con y sin limitador (para graficar)
#   fuera_muestra.csv     escenarios aleatorios que no se usaron al optimizar
#   escalon_tension.csv   respuesta del lazo Q-V a un escalón de consigna (AT-IBR Art. 3-3)
include(joinpath(@__DIR__, "src", "GFM5Barras.jl"))
using .GFM5Barras, Random, Printf

println("Validando con ", Threads.nthreads(), " hilo(s) de Julia")
p = leer_parametros()
s = construir_sistema(p)
dir = joinpath(RAIZ, "resultados")
filas = readlines(joinpath(dir, "optimos.csv"))
cab = split(filas[1], ",")
col(n) = findfirst(==(n), cab)
nθ = ("H1", "H2", "D1", "D2", "Kq1", "Kq2", "K1", "K2")
sint = map(filas[2:end]) do f
    v = split(f, ",")
    (nombre=String(v[col("sintonia")]), modo=String(v[col("modo")]), θ=parse.(Float64, v[[col(n) for n in nθ]]),
     dθ=parse.(Float64, v[[col("d" * n) for n in nθ]]))
end
pfv_nom = p["gfm"]["P_fv"][2]
"Sintonía que aplica en el escenario e (la regla afín depende de la FV disponible, conocida antes del evento)."
θ_de(x, e) = x.θ .+ (pfv_nom - e.pfv[2] * s.S[2]) / 10.0 .* x.dθ
TOL = 1e-3
cs(v) = join(string.(v), ",")
sostenidas = ("th1", "th2", "I1_fin", "I2_fin", "pb1_max", "pb2_max")
columnas(x) = findall(n -> x.modo == "R" || !(n in sostenidas), NOMBRES_G)

# ---------------------------------------------------------------- escenarios de diseño
pares_val = [(x, e, lim) for x in sint for e in s.esc for lim in (0.0, 1.0)]
lin_val = Vector{String}(undef, length(pares_val))
Threads.@threads for idx in eachindex(pares_val)
    x, e, lim = pares_val[idx]
    m, g = evaluar(s, merge(e, (lim=lim,)), θ_de(x, e))
    gp = g[columnas(x)]
    lin_val[idx] = cs([x.nombre, x.modo, e.nombre, Int(lim), m..., g..., max(0.0, maximum(gp)), maximum(gp) <= TOL])
end
open(joinpath(dir, "validacion.csv"), "w") do io
    println(io, cs(["sintonia", "modo", "escenario", "lim", NOMBRES_M..., NOMBRES_G..., "max_violacion", "factible"]))
    foreach(l -> println(io, l), lin_val)
end

# ---------------------------------------------------------------- malla de optimización frente a malla densa
pares_es = [(x, e) for x in sint for e in s.esc]
td = collect(0:0.001:p["optimizacion"]["malla"][end][1])
ig(n) = findfirst(==(n), NOMBRES_G); im(n) = findfirst(==(n), NOMBRES_M)
lin_den = Vector{String}(undef, length(pares_es))
Threads.@threads for idx in eachindex(pares_es)
    x, e = pares_es[idx]
    e0 = merge(e, (lim=x.modo == "L" ? 1.0 : 0.0,))
    m, g = evaluar(s, e0, θ_de(x, e))
    md, gd = evaluar(s, e0, θ_de(x, e); t=td, reltol=1e-10, abstol=1e-12)
    cols = columnas(x)
    lin_den[idx] = cs([x.nombre, x.modo, e.nombre, maximum(g[cols]), maximum(gd[cols]), NOMBRES_G[cols[argmax(gd[cols])]], m[1], md[1],
                       m[im("dfmax_Hz")], md[im("dfmax_Hz")], m[im("I2_pico")], md[im("I2_pico")],
                       1 + g[ig("pb1_max")], 1 + gd[ig("pb1_max")]])
end
open(joinpath(dir, "validacion_densa.csv"), "w") do io
    println(io, cs(["sintonia", "modo", "escenario", "viol_malla", "viol_densa", "restriccion_peor", "J_malla", "J_densa",
                    "dfmax_malla", "dfmax_densa", "I2_pico_malla", "I2_pico_densa", "pb1_pico_malla", "pb1_pico_densa"]))
    foreach(l -> println(io, l), lin_den)
end

# ---------------------------------------------------------------- largo plazo (con limitador)
Ts = p["bess"]["T_servicio"]
tl = collect(0:0.05:Ts)
lin_lp = Vector{String}(undef, length(pares_es))
Threads.@threads for idx in eachindex(pares_es)
    x, e = pares_es[idx]
    θ = θ_de(x, e)
    t, X = simular(s, e, θ; t=tl)
    y = salidas(s, e, t, X)
    m30, _ = evaluar(s, e, θ)
    fm = [min(r.f_sg, r.f1, r.f2) for r in y]
    kmin = argmin(fm)
    pr = [max(r.pred1, r.pred2) for r in y]
    kl = findfirst(>(0.01), pr)
    # segundo descenso: caída de la frecuencia de la SG después de su máximo de recuperación
    k5 = findfirst(>=(5.0), t)
    fsg = [r.f_sg for r in y]
    kmax = k5 - 1 + argmax(fsg[k5:end])
    seg = fsg[kmax] - minimum(fsg[kmax:end])
    lin_lp[idx] = cs([x.nombre, e.nombre, y[findfirst(>=(30.0), t)].f_sg, y[end].f_sg, y[end].I1, y[end].I2,
                      y[end].pb1, y[end].pb2, maximum(r.th1 for r in y) / s.B, maximum(r.th2 for r in y) / s.B,
                      maximum(r.pred1 for r in y), maximum(r.pred2 for r in y), kl === nothing ? NaN : t[kl],
                      y[end].E1, y[end].E2, m30[im("E1_serv_MWs")], m30[im("E2_serv_MWs")],
                      s.f0 - fm[kmin], t[kmin], seg,
                      max(maximum(r.pred1 for r in y) / s.p0[1], maximum(r.pred2 for r in y) / s.p0[2])])
end
open(joinpath(dir, "largo_plazo.csv"), "w") do io
    println(io, cs(["sintonia", "escenario", "f_30", "f_Ts", "I1_Ts", "I2_Ts", "pb1_Ts", "pb2_Ts", "th1_frac", "th2_frac",
                    "pred1_max", "pred2_max", "t_limitador_s", "E1_Ts_real", "E2_Ts_real", "E1_extrapolada",
                    "E2_extrapolada", "dfmax", "t_fmin", "df_segundo_descenso", "pred_max_sobre_p0"]))
    foreach(l -> println(io, l), lin_lp)
end

# ---------------------------------------------------------------- trayectorias (S1)
e_tr = s.esc[1]
t_tr = collect(0:0.02:60)
pares_tr = [(x, lim) for x in sint for lim in (0.0, 1.0)]
res_tr = Vector{Any}(undef, length(pares_tr))
Threads.@threads for idx in eachindex(pares_tr)
    x, lim = pares_tr[idx]
    t, X = simular(s, merge(e_tr, (lim=lim,)), θ_de(x, e_tr); t=t_tr)
    res_tr[idx] = (x.nombre, Int(lim), salidas(s, e_tr, t, X))
end
open(joinpath(dir, "trayectorias.csv"), "w") do io
    cabt = false
    for (nom, lim_i, ys) in res_tr, r in ys
        cabt || (println(io, cs(["sintonia", "lim", string.(keys(r))...])); cabt = true)
        println(io, cs([nom, lim_i, values(r)...]))
    end
end

# ---------------------------------------------------------------- fuera de muestra
rng = MersenneTwister(2026)
N = 500
muestras = [Dict("nombre" => "m$k", "barra" => rand(rng, 2:5), "dP" => 20 + 30 * rand(rng),
                 "SOC0" => 0.2 .+ 0.6 .* rand(rng, 2), "P_fv2" => 35 + 10 * rand(rng)) for k in 1:N]
escs_m = [s.nuevo_esc(d) for d in muestras]
pares_fm = [(k, j) for k in 1:N for j in eachindex(sint)]
lin_fm = Vector{String}(undef, length(pares_fm))
i_pred1, i_pred2 = im("pred1_max"), im("pred2_max")
cols_duro = findall(n -> n in ("I1_max", "I2_max", "df_max", "rocof", "V_min", "V_max") ||
                       startswith(n, "ib") || startswith(n, "vbat") || startswith(n, "SOC"), NOMBRES_G)
Threads.@threads for idx in eachindex(pares_fm)
    k, j = pares_fm[idx]
    d, e, x = muestras[k], escs_m[k], sint[j]
    θ = θ_de(x, e)
    m, g = evaluar(s, sin_limitador(e), θ)
    ml, gl = evaluar(s, merge(e, (lim=1.0,)), θ)
    lin_fm[idx] = cs([x.nombre, x.modo, d["nombre"], d["barra"], d["dP"], d["SOC0"]..., d["P_fv2"], m[1], m[2],
                      max(0.0, maximum(g)), NOMBRES_G[argmax(g)], maximum(g) <= TOL, ml[1], ml[2],
                      max(ml[i_pred1], ml[i_pred2]), e.pb0[2], maximum(gl[cols_duro]) <= TOL,
                      max(ml[i_pred1], ml[i_pred2]) > 0.01])
end
open(joinpath(dir, "fuera_muestra.csv"), "w") do io
    println(io, cs(["sintonia", "modo", "muestra", "barra", "dP", "SOC1", "SOC2", "P_fv2", "J", "dfmax_Hz", "max_violacion",
                    "restriccion_peor", "factible", "J_lim", "dfmax_lim", "pred_max", "pb0_2", "factible_duro", "activa_limitador"]))
    foreach(l -> println(io, l), lin_fm)
end

# ---------------------------------------------------------------- escalón de consigna de tensión (1 %)
e_esc = merge(s.esc[1], (Yred=s.esc[1].Yred_pre, W=s.esc[1].W_pre))        # red previa al evento (equilibrio)
t_esc = collect(0:0.001:10)
pares_esc = [(x, i) for x in sint for i in 1:2]
lin_esc = Vector{String}(undef, length(pares_esc))
Threads.@threads for idx in eachindex(pares_esc)
    x, i = pares_esc[idx]
    U0 = copy(s.U0); U0[i] *= 1.01
    _, X = simular(merge(s, (U0=U0,)), e_esc, x.θ; t=t_esc)
    y = salidas(s, e_esc, t_esc, X)
    u = [i == 1 ? r.U1 : r.U2 for r in y]
    Δ = u .- u[1]
    fin = Δ[end]
    cruce(f) = t_esc[findfirst(k -> Δ[k] / fin >= f, eachindex(t_esc))]
    fuera = findlast(k -> abs(Δ[k] / fin - 1) > 0.05, eachindex(t_esc))
    lin_esc[idx] = cs([x.nombre, i, cruce(0.1), cruce(0.9) - cruce(0.1), fuera === nothing ? 0.0 : t_esc[min(fuera + 1, end)],
                       100 * (maximum(Δ ./ fin) - 1)])
end
open(joinpath(dir, "escalon_tension.csv"), "w") do io
    println(io, cs(["sintonia", "gfm", "t_reaccion_s", "t_crecimiento_s", "t_establecimiento_s", "sobreoscilacion_pct"]))
    foreach(l -> println(io, l), lin_esc)
end
println("Validación escrita en ", dir)
