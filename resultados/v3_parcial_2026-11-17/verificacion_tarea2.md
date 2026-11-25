# Verificación de cierre de Tarea 2

PDF: 7 páginas. PEND: 0. CSV exigidos ausentes: ninguno.
Compilación: pdflatex, bibtex, pdflatex, pdflatex; sin referencias sin resolver ni overfull.
Introducción: 12 claves bibliográficas. Tabla de cierre: 53 ítems.
No se relanzaron experimentos; se ejecutaron graficar.py y resumen.py.

## Última línea de cada log

- `ablacionn6.log`: [11:59:37] Listo: ablacion
- `barridosn3.log`: [12:26:48] Listo: barridos
- `cert.log`: [13:31:50] Listo: cert
- `codex_tarea1.log`: Observación: el pie de convergencia dice 0,1 %; la tabla usa el 0,5 % solicitado. Dejé intacto el texto del informe.
- `convmetodoF1n10.log`: [11:22:38] Listo: conv
- `convmetodoF2n20.log`: [11:17:08] Listo: conv
- `convmetodoF3n20.log`: [12:12:15] Listo: conv
- `convmetodoF4bn4.log`: [11:17:50] Listo: conv
- `convmetodoF4n20.log`: [11:45:25] Listo: conv
- `convmetodoF5n10.log`: [13:06:56] Listo: conv
- `malla.log`: [12:53:44] Listo: malla
- `modosn6.log`: [11:25:32] Listo: modos
- `optimos.log`: [13:24:52] Listo: optimos
- `orquestador.log`: (vacío)
- `pesosn6.log`: [12:09:41] Listo: pesos
- `robuston6.log`: [13:24:52] Listo: robusto
- `validar.log`: Validación escrita en /home/jay_c/Documents/Universidad/Opti Avz/Proyecto/Codigos_final/resultados

## Logs sin «Listo» y advertencias

codex_tarea1.log: reporte anterior; orquestador.log: vacío; validar.log: termina «Validación escrita…».
No se encontró una excepción fatal de experimento. cert.log advierte Solution may be inaccurate;
robusto.csv registra Maximum_CpuTime_Exceeded para minmax_F4.

## Contradicciones corregidas (fuentes: CSV de resultados/)

- modos.csv: H/D interiores en R de SQP, pero H1/D1 en cotas en L; Kq mínimo en ambos.
- optimos.csv y validacion.csv: R consolidado es F4 (J=0,63357 sin limitador), no SQP (0,63164).
- validacion.csv: F1 tiene J=0,64902 sin limitador, mayor que R consolidado.
- robusto.csv: min–max/esperado/CVaR son infactibles (~3,33 %).
- validacion_densa.csv: regla afín viola 0,2523 % en S2.
- certificados.csv: regla afín tiene amortiguamiento mínimo 4,82 % y tasa común -inf; no hay certificado aceptado.
- barridos.csv: multiplicador y diferencias discrepan 79,85 % a reserva 5 %; precio sombra no validado.
- largo_plazo.csv: no se agota el presupuesto térmico ni aparece segundo descenso apreciable a 300 s.
- corridas_F*.csv: F1/F4b/F5 sólo tienen 10/4/10 corridas; F2/F3/F4 tienen 20.
- Validación densa y fuera de muestra sólo revisan trayectorias sin limitador, no (P) completo.

Resumen numérico guardado en resumen_tarea2.txt. Pendientes en CONTINUAR.md §4 y Lista_de_mejoras.md.
