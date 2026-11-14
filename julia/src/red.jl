# Red eléctrica: matriz de admitancia nodal (modelo π) y reducción de Kron.

"Ybus [pu] a partir de los parámetros de secuencia positiva por km de cada línea."
function ybus_lineas(p)
    f0, Sb, Vb = p["sistema"]["f0"], p["sistema"]["Sb"], p["sistema"]["Vb"]
    Zb, w = Vb^2 / Sb, 2π * f0
    lineas = p["red"]["lineas"]
    n = maximum(max(Int(l[1]), Int(l[2])) for l in lineas)
    Y = zeros(ComplexF64, n, n)
    for l in lineas
        i, j = Int(l[1]), Int(l[2])
        largo, R, L, C = l[3], l[4], l[5], l[6]
        y = 1 / ((R + im * w * L) * largo / Zb)    
        bsh = w * C * largo * Zb                    # Aquí utilizamos la capacitancia total y posteriormente pasamos al 
        Y[i, i] += y + im * bsh / 2                 # cálculo de la matriz de admitancia nodal. 
        Y[j, j] += y + im * bsh / 2                 # La capacitancia se reparte entre las dos barras.
        Y[i, j] -= y
        Y[j, i] -= y
    end
    return Y
end

"""
Reduce la red a los nodos internos de las fuentes (fem detrás de reactancia).
`yL`: admitancias shunt de carga por barra; `barras`, `X`: barra y reactancia [pu sistema] de cada fuente.
Devuelve Yred (I = Yred*E) y W (V_barras = W*E).
"""
function kron(Ybus, yL, barras, X)
    nb, ng = size(Ybus, 1), length(barras)
    Ybb = Ybus + Diagonal(yL)
    Yib = zeros(ComplexF64, ng, nb)
    for k in 1:ng
        y = 1 / (im * X[k])
        Ybb[barras[k], barras[k]] += y
        Yib[k, barras[k]] = -y
    end
    Yii = Diagonal([1 / (im * X[k]) for k in 1:ng])
    W = -(Ybb \ transpose(Yib))
    return Matrix(Yii + Yib * W), W
end
