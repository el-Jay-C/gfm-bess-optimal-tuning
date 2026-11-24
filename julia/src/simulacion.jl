# Simulación no lineal y cálculo de métricas de desempeño y restricciones.

# Restricciones g ≤ 0, todas normalizadas por su límite (violación relativa).
const NOMBRES_G = ["I1_max", "I2_max", "th1", "th2", "I1_fin", "I2_fin", "ib1_max", "ib2_max",
                   "vbat1_min", "vbat2_min", "pb1_max", "pb2_max", "df_max", "rocof",
                   "V_banda", "E1_serv", "E2_serv", "SOC1_min", "SOC2_min", "SOC1_max", "SOC2_max", "pb1_sost", "pb2_sost"]
const NOMBRES_M = ["J", "dfmax_Hz", "fmin_Hz", "t_fmin_s", "vdev", "perdidas", "vdev_paper", "p1_max", "p2_max",
                   "rocof_Hz_s", "V_min", "V_max", "I1_pico", "I2_pico", "I1_fin", "I2_fin", "th1_frac", "th2_frac",
                   "E1_serv_MWs", "E2_serv_MWs", "pred1_max", "pred2_max", "f_fin_Hz",
                   "alivio_carga", "pb1_fin", "pb2_fin", "pb1_pico", "pb2_pico"]

"Malla de tiempos (no uniforme) de parametros.toml. `factor` > 1 la refina."
function malla(p; factor=1)
    t = [0.0]
    for (tf, h) in p["optimizacion"]["malla"]
        n = round(Int, (tf - t[end]) / h) * factor
        append!(t, range(t[end], tf; length=n + 1)[2:end])
    end
    return t
end

function simular(s, e, θ; t=malla(s.p), reltol=1e-8, abstol=1e-10)
    prob = ODEProblem(rhs!, copy(e.x0), (0.0, t[end]), (s, e, θ))
    sol = solve(prob, Rodas5P(); saveat=t, abstol=abstol, reltol=reltol, maxiters=1_000_000)
    SciMLBase.successful_retcode(sol) || return nothing
    return sol.t, reduce(hcat, sol.u)
end

"Trayectorias de todas las salidas de interés (una fila por instante)."
function salidas(s, e, t, X)
    f0 = s.f0
    map(eachindex(t)) do k
        x = X[:, k]
        a = algebra(x, s, e)
        Vb = abs.(a.V)
        pcarga = sum(e.GL .* Vb .^ 2) + sum(real, e.Sc)
        ploss = sum(real, a.Sint) - pcarga
        (t=t[k], f_sg=f0 * x[2], f1=f0 * x[igfm(1)+2], f2=f0 * x[igfm(2)+2],
         I1=a.g[1].iac, I2=a.g[2].iac, p1=a.g[1].p, p2=a.g[2].p, q1=a.g[1].q, q2=a.g[2].q,
         U1=a.g[1].U, U2=a.g[2].U, pb1=a.g[1].pb, pb2=a.g[2].pb, ib1=a.g[1].ib, ib2=a.g[2].ib,
         vbat1=a.g[1].vbat, vbat2=a.g[2].vbat, SOC1=x[igfm(1)+5], SOC2=x[igfm(2)+5],
         E1=x[igfm(1)+7], E2=x[igfm(2)+7], th1=x[igfm(1)+8], th2=x[igfm(2)+8],
         pred1=x[igfm(1)+9] + s.kp_lim * exceso_lim(a.g[1], s, 1),
         pred2=x[igfm(2)+9] + s.kp_lim * exceso_lim(a.g[2], s, 2), ilim1=a.g[1].ilim, ilim2=a.g[2].ilim,
         pg=a.pg, ploss=ploss, pcarga=pcarga, V1=Vb[1], V2=Vb[2], V3=Vb[3], V4=Vb[4], V5=Vb[5])
    end
end

"Integral por trapecios sobre una malla no uniforme."
trapz(y, t) = sum((y[k] + y[k+1]) / 2 * (t[k+1] - t[k]) for k in 1:length(t)-1)

"Objetivo, métricas y restricciones agregadas g ≤ 0 (definiciones idénticas a CasADi)."
function metricas(s, e, t, X)
    p, n, b, gf = s.p, s.p["norma"], s.p["bess"], s.p["gfm"]
    o = p["optimizacion"]
    T = t[end]
    y = salidas(s, e, t, X)
    col(c) = [getfield(r, c) for r in y]
    F = hcat(col(:f_sg), col(:f1), col(:f2))
    dfmax = maximum(abs.(F .- s.f0))
    w = n["ventana_rocof"]
    rocof = maximum(abs(F[k, j] - F[max(searchsortedlast(t, t[k] - w + 1e-9), 1), j]) / w
                    for k in eachindex(t), j in 1:3)
    Vb = hcat(col(:V1), col(:V2), col(:V3), col(:V4), col(:V5))
    Vpre = abs.(s.V)
    vdev = trapz(vec(sum(((Vb .- Vpre') ./ o["dV_ref"]) .^ 2, dims=2)), t) / (T * 5)
    vpap = trapz(vec(sum(((Vb .- 1) ./ (n["V_max_alerta"] - n["V_min_alerta"])) .^ 2, dims=2)), t) / (T * 5)
    perd = trapz(col(:ploss) .- s.ploss0, t) / (T * e.dP)       # aumento de pérdidas respecto del punto previo
    J = o["wf"] * dfmax / o["df_ref"] + o["wv"] * vdev + o["wl"] * perd
    I1, I2 = col(:I1), col(:I2)
    # Energía que exige sostener la respuesta hasta T_servicio (se extrapola la potencia final del BESS)
    Ts = b["T_servicio"]
    Eserv = [col(:E1)[end] + (Ts - T) * (col(:pb1)[end] - e.pb0[1]) * s.Pn[1],
             col(:E2)[end] + (Ts - T) * (col(:pb2)[end] - e.pb0[2]) * s.Pn[2]]
    g = [maximum(I1) / s.Imax - 1, maximum(I2) / s.Imax - 1,
         maximum(col(:th1)) / s.B - 1, maximum(col(:th2)) / s.B - 1,
         I1[end] / s.Icont - 1, I2[end] / s.Icont - 1,
         maximum(abs, col(:ib1)) / b["i_max"] - 1, maximum(abs, col(:ib2)) / b["i_max"] - 1,
         1 - minimum(col(:vbat1)) / b["v_min"], 1 - minimum(col(:vbat2)) / b["v_min"],
         maximum(abs, col(:pb1)) - 1, maximum(abs, col(:pb2)) - 1,
         dfmax / (s.f0 - n["f_min_perm"]) - 1, rocof / n["rocof_max"] - 1,
         maximum(abs.(Vb .- 1)) / (n["V_max_normal"] - 1) - 1,
         Eserv[1] / s.Eres[1] - 1, Eserv[2] / s.Eres[2] - 1,
         b["SOC_min"] - minimum(col(:SOC1)), b["SOC_min"] - minimum(col(:SOC2)),
         maximum(col(:SOC1)) - b["SOC_max"], maximum(col(:SOC2)) - b["SOC_max"],
         col(:pb1)[end] - 1, col(:pb2)[end] - 1]
    fmin, kmin = findmin(vec(minimum(F, dims=2)))
    m = [J, dfmax, fmin, t[kmin], vdev, perd, vpap, maximum(col(:p1)), maximum(col(:p2)), rocof,
         minimum(Vb), maximum(Vb), maximum(I1), maximum(I2), I1[end], I2[end],
         maximum(col(:th1)) / s.B, maximum(col(:th2)) / s.B, Eserv[1], Eserv[2],
         maximum(col(:pred1)), maximum(col(:pred2)), F[end, 1],
         # fracción del escalón que no se consume en el nadir porque la carga (impedancia constante) baja con V²
         1 - (col(:pcarga)[kmin] - (sum(real, s.Sgen) - s.ploss0)) / e.dP,
         col(:pb1)[end], col(:pb2)[end], maximum(col(:pb1)), maximum(col(:pb2))]
    return m, g, y
end

"Evalúa θ en el escenario e: devuelve (métricas, g). Si la simulación falla, valores de castigo."
function evaluar(s, e, θ; kw...)
    r = simular(s, e, θ; kw...)
    r === nothing && return fill(1e3, length(NOMBRES_M)), fill(1e3, length(NOMBRES_G))
    m, g, _ = metricas(s, e, r...)
    return m, g
end

"Copia del escenario con el limitador de sobrecarga deshabilitado (modelo del artículo base)."
sin_limitador(e) = merge(e, (lim=0.0,))
