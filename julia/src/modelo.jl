# Modelo dinámico fasorial (RMS) del sistema de 5 barras
#
# - Máquina síncrona: modelo clásico con gobernador y turbina con recalentador.
# - 2 inversores GFM tipo VSG: ecuación de oscilación virtual (H, D) y lazo Q-V integral (Kq, K) con filtro de
#   medición de Q, reactancia de acoplamiento y pérdidas del convertidor.
# - Limitador de sobrecarga de cada GFM: presupuesto térmico I²t (ϑ) y reducción de la consigna de potencia (pred)
#   cuando la corriente supera el límite vigente (I_max con presupuesto disponible, I_cont al agotarse) o cuando
#   el BESS supera su potencia nominal.
# - Lado DC: BESS como circuito de Thevenin de 1er orden, SOC con eficiencia coulómbica y energía entregada.
#
# Estados (22): SG [δ, ω, xg, xr] + por GFM [δ, ω, E, qm, SOC, vrc, e, ϑ, pred].
# Parámetros θ: [H1 H2 D1 D2 Kq1 Kq2 K1 K2].

const NG = 9                     # estados por GFM
const NX = 4 + 2NG
const N_PF = 12                  # iteraciones de punto fijo de la carga a potencia constante (igual en CasADi)
igfm(i) = 4 + NG * (i - 1)

"Absoluto suave (para la eficiencia de carga, idéntico en Julia y CasADi)."
suave_abs(x) = sqrt(x^2 + 1e-6)
"Rampa suave ≈ max(x, 0) de ancho ε (softplus en forma numéricamente estable)."
rampa(x, ε) = max(x, zero(x)) + ε * log1p(exp(-abs(x) / ε))
"Sigmoide logística (forma con tanh, sin desbordes)."
sigm(z) = (1 + tanh(z / 2)) / 2

"Construye el sistema: flujo de potencia, condiciones iniciales y red reducida por escenario."
function construir_sistema(p)
    Sb, f0 = p["sistema"]["Sb"], p["sistema"]["f0"]
    sg, gf, bs, red = p["sg"], p["gfm"], p["bess"], p["red"]
    Y = ybus_lineas(p)
    nb = size(Y, 1)
    PL, QL = red["carga_P"] ./ Sb, red["carga_Q"] ./ Sb

    # Flujo de potencia: barra SG slack, barras GFM PV, resto PQ
    tipo = fill(3, nb); tipo[sg["barra"]] = 1
    V0 = ones(ComplexF64, nb); V0[sg["barra"]] = red["V_slack"]
    Pg, Qg = zeros(nb), zeros(nb)
    for i in 1:2
        b = gf["barra"][i]
        tipo[b] = 2; V0[b] = gf["V0"][i]; Pg[b] = gf["P0"][i] / Sb
    end
    V, S, it = flujo_potencia(Y, tipo, V0, Pg - PL, Qg - QL)
    Sgen = S + PL + im * QL

    # SG y GFM como fem detrás de reactancia [pu del sistema]
    barras = [sg["barra"]; gf["barra"]]
    Sbase = [sg["S"]; gf["S"]]
    X = [sg["xd_p"]; gf["Xf"]] .* Sb ./ Sbase
    Ifu = conj.(Sgen[barras] ./ V[barras])
    E = V[barras] + im * X .* Ifu

    pac = real.(Sgen[gf["barra"]]) .* Sb ./ gf["S"]
    iac = abs.(Ifu[2:3]) .* Sb ./ gf["S"]
    Rt = bs["R0"] + bs["R1"]
    frac = red["frac_P"]                                    # fracción a potencia constante
    yL0 = (1 - frac) .* (PL - im * QL) ./ abs2.(V)
    Sc0 = frac .* (PL + im * QL)
    Yred_pre, W_pre, Z_pre = kron(Y, yL0, barras, X)

    "Escenario: escalón de carga dP [MW] en `barra`, SOC inicial y FV disponible en el GFM2 [MW]."
    function nuevo_esc(e)
        yL, Sc = copy(yL0), copy(Sc0)
        yL[e["barra"]] += (1 - frac) * e["dP"] / Sb / abs2(V[e["barra"]])
        Sc[e["barra"]] += frac * e["dP"] / Sb
        Yred, W, Z = kron(Y, yL, barras, X)
        pfv = [gf["P_fv"][1], get(e, "P_fv2", gf["P_fv"][2])] ./ gf["S"]
        pb0 = (pac + gf["R_conv"] .* iac .^ 2 - pfv) .* gf["S"] ./ bs["P_nom"]     # punto de operación del BESS
        x0 = zeros(NX)
        x0[1] = angle(E[1]); x0[2] = 1.0
        for i in 1:2
            k = igfm(i); soc = e["SOC0"][i]
            ocv = bs["ocv0"] + bs["ocv1"] * soc
            ib = (ocv - sqrt(ocv^2 - 4Rt * pb0[i])) / (2Rt)
            x0[k+1:k+7] = [angle(E[i+1]), 1.0, abs(E[i+1]),
                           imag(V[barras[i+1]] * conj(Ifu[i+1])) * Sb / gf["S"][i],
                           soc, bs["R1"] * ib, 0.0]
        end
        (nombre=e["nombre"], Yred=Yred, W=W, GL=real.(yL), x0=x0, pfv=pfv, pb0=pb0,
         Z=Z, Sc=Sc, V0=V, Yred_pre=Yred_pre, W_pre=W_pre, Z_pre=Z_pre, Sc_pre=Sc0, dP=e["dP"] / Sb, lim=1.0)
    end
    esc = map(nuevo_esc, p["escenario"])

    B = (gf["I_max"]^2 - gf["I_cont"]^2) * gf["t_sobrecorriente"]
    Eres = bs["reserva"] .* bs["P_nom"] .* bs["E_h"] .* 3600.0                     # MWs
    return (p=p, Sb=Sb, f0=f0, wb=2π * f0, V=V, Sgen=Sgen, it_fp=it, X=X, E0=E, Ifu=Ifu,
            barras=barras, Sg=sg["S"], Hg=sg["H"], Dg=sg["D"], Rg=sg["R"], TG=sg["TG"],
            TR=sg["TR"], FH=sg["FH"], Eg=abs(E[1]), pm0=real(Sgen[sg["barra"]]) * Sb / sg["S"],
            S=Float64.(gf["S"]), Xf=X[2:3], tau_q=Float64.(gf["tau_q"]), Rconv=Float64.(gf["R_conv"]),
            p0=pac, q0=[esc[1].x0[igfm(i)+4] for i in 1:2], U0=abs.(V[gf["barra"]]), i0=iac,
            Pn=Float64.(bs["P_nom"]), Eh=Float64.(bs["E_h"]), ocv0=bs["ocv0"], ocv1=bs["ocv1"],
            R0=bs["R0"], R1=bs["R1"], tau1=bs["tau1"], eta=bs["eta_c"], Eres=Eres,
            Imax=gf["I_max"], Icont=gf["I_cont"], B=B, klim=gf["k_lim"], kp_lim=gf["kp_lim"], tau_lib=gf["tau_lib"],
            eps_i=gf["eps_i"], eps_th=gf["eps_th"], mth=gf["margen_th"], ath=gf["ancho_th"],
            ploss0=sum(real, Sgen) - sum(PL), esc=esc, nuevo_esc=nuevo_esc)
end

"Límite de corriente vigente según el presupuesto térmico consumido ϑ."
ilimite(ϑ, s) = s.Icont + (s.Imax - s.Icont) * sigm((s.mth * s.B - ϑ) / (s.ath * s.B))

"Variables algebraicas (red + convertidor + banco) para un estado x."
function algebra(x, s, e)
    T = eltype(x)
    E = Complex{T}[s.Eg * cis(x[1]); [x[igfm(i)+3] * cis(x[igfm(i)+1]) for i in 1:2]]
    V = e.W * E
    Iiny = zeros(Complex{T}, length(V))
    if any(!iszero, e.Sc)                                    # carga a potencia constante: punto fijo en V
        for _ in 1:N_PF
            Iiny = -conj.(e.Sc ./ V)
            V = e.W * E + e.Z * Iiny
        end
    end
    I = e.Yred * E - transpose(e.W) * Iiny
    Sint = E .* conj.(I)
    pg = real(Sint[1]) * s.Sb / s.Sg
    out = map(1:2) do i
        k, f = igfm(i), s.Sb / s.S[i]
        Ii = I[i+1]
        Vt = E[i+1] - im * s.Xf[i] * Ii
        p = real(Sint[i+1]) * f
        q = imag(Vt * conj(Ii)) * f
        iac = abs(Ii) * f
        pb = (p + s.Rconv[i] * iac^2 - e.pfv[i]) * s.S[i] / s.Pn[i]
        vo = s.ocv0 + s.ocv1 * x[k+5] - x[k+6]
        ib = (vo - sqrt(max(vo^2 - 4s.R0 * pb, 1e-9))) / (2s.R0)
        (p=p, q=q, U=abs(Vt), iac=iac, pb=pb, ib=ib, vbat=vo - s.R0 * ib, ilim=ilimite(x[k+8], s))
    end
    return (E=E, I=I, Sint=Sint, V=V, pg=pg, g=out)
end

"Exceso de corriente y de potencia del BESS [pu del inversor] que ve el limitador."
exceso_lim(g, s, i) = rampa(g.iac - g.ilim, s.eps_i) + s.Pn[i] / s.S[i] * rampa(g.pb - 1, s.eps_i)

"Lado derecho dx/dt = f(x, θ) para el escenario e."
function rhs!(dx, x, (s, e, θ), t)
    a = algebra(x, s, e)
    wg = x[2]
    pm = s.pm0 + x[4] + s.FH * (x[3] - x[4])
    dx[1] = s.wb * (wg - 1)
    dx[2] = (pm - a.pg - s.Dg * (wg - 1)) / (2s.Hg)
    dx[3] = (-(wg - 1) / s.Rg - x[3]) / s.TG
    dx[4] = (x[3] - x[4]) / s.TR
    for i in 1:2
        k, g = igfm(i), a.g[i]
        H, D, Kq, K = θ[i], θ[2+i], θ[4+i], θ[6+i]
        exceso = exceso_lim(g, s, i)
        pred = e.lim * (x[k+9] + s.kp_lim * exceso)              # reducción de consigna del limitador (PI)
        dx[k+1] = s.wb * (x[k+2] - 1)
        dx[k+2] = (s.p0[i] - pred - g.p - D * (x[k+2] - 1)) / (2H)
        dx[k+3] = K * (Kq * (s.U0[i] - g.U) + s.q0[i] - x[k+4])
        dx[k+4] = (g.q - x[k+4]) / s.tau_q[i]
        ieff = 0.5 * (1 + s.eta) * g.ib + 0.5 * (1 - s.eta) * suave_abs(g.ib)
        dx[k+5] = -ieff / (3600 * s.Eh[i])
        dx[k+6] = (s.R1 * g.ib - x[k+6]) / s.tau1
        dx[k+7] = (g.pb - e.pb0[i]) * s.Pn[i]                    # energía entregada sobre el punto previo [MWs]
        dx[k+8] = rampa(g.iac^2 - s.Icont^2, s.eps_th)           # presupuesto térmico consumido [pu²·s]
        # Limitador: integra el exceso de corriente sobre el límite vigente y el exceso de potencia del BESS
        # sobre su potencia nominal; se libera cuando ambos vuelven bajo su límite.
        dx[k+9] = s.klim * exceso -
                  x[k+9] / s.tau_lib * sigm((g.ilim - g.iac) / s.eps_i) * sigm((1 - g.pb) / s.eps_i)
    end
    return nothing
end
