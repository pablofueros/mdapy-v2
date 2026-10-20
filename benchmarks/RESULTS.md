# Resultados

## Entorno y protocolo

Medición realizada el **20 de octubre de 2026** en Linux sobre WSL2, con
arquitectura ARM64 y 12 CPU lógicas. Se utilizaron Python 3.12.3, NumPy 1.26.0
y SciPy 1.15.1.

La carga contiene las 140.256 filas de `datawave.npz`, con las variables `hs`,
`tp` y `dir` en `float64`. La semilla es la observación con mayor `tp`, índice
27359. Se midieron 5, 10 y 20 selecciones, con cinco repeticiones en procesos
independientes y un límite de 180 segundos por proceso.

Los datos se cargan antes de preparar cada referencia. `preparación` incluye
conversión o copia de columnas, desviaciones y matriz de trabajo; `selección`
empieza con los datos ya preparados. `llamada` es la suma de ambas y `proceso`
incluye además arranque de Python, imports y carga. La memoria es el RSS máximo
del proceso completo.

## Resultados

| Selecciones | Versión | Preparación, mediana (s) | Selección, mediana (s) | Llamada, mediana (s) | Proceso, media (s) | RSS máximo, mediana (MiB) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 5 | 2.0 | 0,0045 | 5,3295 | 5,3376 | 5,8689 | 189,6 |
| 5 | 2.1 | 0,0060 | 0,0219 | 0,0306 | 0,3872 | 114,9 |
| 10 | 2.0 | 0,0047 | 13,0772 | 13,0819 | 13,6912 | 293,6 |
| 10 | 2.1 | 0,0074 | 0,0391 | 0,0443 | 0,4231 | 112,9 |
| 20 | 2.0 | 0,0045 | 32,6276 | 32,6321 | 33,5223 | 507,1 |
| 20 | 2.1 | 0,0078 | 0,0772 | 0,0864 | 0,4542 | 118,1 |

Las dos versiones eligieron los mismos índices y en el mismo orden para todas
las cargas de `datawave`. Con veinte selecciones fueron:

```text
[27359, 128807, 123587, 55022, 123571, 63238, 11672, 32875, 62341,
 72553, 121178, 118697, 79858, 27814, 16560, 25991, 30794, 88931,
 23257, 121744]
```

## Interpretación

La mejora está en la selección, no en la preparación. Para veinte selecciones,
la llamada completa de 2.1 fue aproximadamente 378 veces más rápida y la
selección aislada unas 423 veces más rápida en esta máquina. El tiempo total
medio del proceso se redujo de 33,52 a 0,45 segundos.

La diferencia aumenta con el número de selecciones, de acuerdo con la evolución
de un recorrido repetido del subconjunto a mínimos incrementales. Al pasar de
5 a 20 selecciones, la mediana de la llamada creció unas 6,1 veces en 2.0 y
unas 2,8 veces en 2.1. El RSS máximo mediano de 2.0 pasó de 189,6 a 507,1 MiB,
mientras que 2.1 se mantuvo entre 112,9 y 118,1 MiB. Con veinte selecciones,
el RSS observado en 2.1 fue un 77 % menor.

Los casos pequeños también pasaron. Existe una diferencia intencionada en el
caso de casi empate: 2.0 selecciona `[0, 1, 3]` después de convertir la entrada
a `float32`, mientras que 2.1 selecciona `[0, 2, 3]` al conservar `float64`.
Por tanto, la equivalencia observada en `datawave` no implica equivalencia para
todos los datos y tipos posibles.

Estos valores describen este dataset, este entorno y estas cargas. El RSS
incluye Python, NumPy, SciPy y los datos, por lo que no aísla por sí solo cada
asignación del algoritmo. Los tiempos tampoco deben extrapolarse directamente
a valores grandes de `k` sin medirlos.
