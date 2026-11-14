# Flujo de potencia AC por Newton-Raphson en coordenadas polares.
# tipo: 1 = slack, 2 = PV, 3 = PQ. P, Q: inyecciones especificadas [pu].

function flujo_potencia(Y, tipo, V0, P, Q; tol=1e-10, itmax=30)
    Vm, θ = abs.(V0), angle.(V0)
    pv, pq = findall(==(2), tipo), findall(==(3), tipo)
    ns = [pv; pq]
    for it in 1:itmax
        V = Vm .* cis.(θ)
        I = Y * V
        S = V .* conj.(I)
        F = [P[ns] - real(S[ns]); Q[pq] - imag(S[pq])]
        maximum(abs, F) < tol && return V, S, it
        dSdθ = im * Diagonal(V) * conj.(Diagonal(I) - Y * Diagonal(V))    # Definimos las derivadas con respecto a θ y |V| en forma matricial
        Vn = V ./ Vm
        dSdV = Diagonal(V) * conj.(Y * Diagonal(Vn)) + conj.(Diagonal(I)) * Diagonal(Vn)
        J = [real(dSdθ[ns, ns]) real(dSdV[ns, pq]);
             imag(dSdθ[pq, ns]) imag(dSdV[pq, pq])]
        dx = J \ F
        θ[ns] .+= dx[1:length(ns)]
        Vm[pq] .+= dx[length(ns)+1:end]
    end
    error("El flujo de potencia no convergió en $itmax iteraciones")
end
