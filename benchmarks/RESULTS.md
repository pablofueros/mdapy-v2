# Resultados

## 2.0 frente a 2.1

Se comparan `v2.0.0` y `v2.1.0`, los últimos parches disponibles de ambas
evoluciones.

### Qué cambió

La versión 2.0 convierte los datos a `float32`. Para elegir cada observación,
recorre todos los candidatos, calcula su distancia al subconjunto seleccionado
y conserva una caché de pares.

La versión 2.1 preserva el dtype común de NumPy, calcula las distancias por
lotes y mantiene únicamente la distancia mínima de cada candidato. El trabajo
de selección pasa aproximadamente de `O(n * k**2)` a `O(n * k)` y desaparece
la caché que crecía con el número de selecciones.

Ambas evoluciones seleccionan de forma iterativa el punto más alejado de su
punto seleccionado más cercano. Las distancias se normalizan por la desviación
de cada variable y los empates exactos favorecen el menor índice. El factor
global `sqrt(2)` presente en 2.0 desaparece en 2.1; matemáticamente no altera el
orden de las distancias.

Los casos de referencia pasaron. En el caso de casi empate, 2.0 produce
`[0, 1, 3]` al convertir a `float32`, mientras que 2.1 produce `[0, 2, 3]` al
conservar `float64`. Es una diferencia esperada, no una regresión.

### Experimento

Medición del **20 de octubre de 2026** en Linux sobre WSL2 ARM64, con 12 CPU
lógicas, Python 3.12.3, NumPy 1.26.0 y SciPy 1.15.1.

Se utilizaron las 140.256 filas de `datawave.npz`, las variables `hs`, `tp` y
`dir` en `float64`, y la observación con mayor `tp` como semilla (índice 27359).
Se midieron 5, 10 y 20 selecciones, con cinco procesos independientes por carga
y un límite de 180 segundos por proceso.

`Preparación` incluye conversiones o copias, desviaciones y matriz de trabajo.
`Selección` comienza con esos datos preparados. `Llamada` suma ambos tiempos;
`proceso` incluye además arranque, imports y carga. El RSS corresponde al
proceso completo.

| Selecciones | Versión | Preparación, mediana (s) | Selección, mediana (s) | Llamada, mediana (s) | Proceso, media (s) | RSS máximo, mediana (MiB) |
| ----------: | :------ | -----------------------: | ---------------------: | -------------------: | -----------------: | ------------------------: |
|           5 | 2.0     |                   0,0045 |                 5,3295 |               5,3376 |             5,8689 |                     189,6 |
|           5 | 2.1     |                   0,0060 |                 0,0219 |               0,0306 |             0,3872 |                     114,9 |
|          10 | 2.0     |                   0,0047 |                13,0772 |              13,0819 |            13,6912 |                     293,6 |
|          10 | 2.1     |                   0,0074 |                 0,0391 |               0,0443 |             0,4231 |                     112,9 |
|          20 | 2.0     |                   0,0045 |                32,6276 |              32,6321 |            33,5223 |                     507,1 |
|          20 | 2.1     |                   0,0078 |                 0,0772 |               0,0864 |             0,4542 |                     118,1 |

Las dos versiones eligieron los mismos índices de `datawave` en todas las
repeticiones.

### Lectura

Con veinte selecciones, la llamada de 2.1 fue unas 378 veces más rápida y la
selección aislada unas 423 veces más rápida. El tiempo medio del proceso bajó
de 33,52 a 0,45 segundos.

Al pasar de 5 a 20 selecciones, la llamada creció unas 6,1 veces en 2.0 y 2,8
veces en 2.1. El RSS mediano de 2.0 pasó de 189,6 a 507,1 MiB; en 2.1 permaneció
entre 112,9 y 118,1 MiB. Para veinte selecciones fue un 77 % menor.

Los resultados demuestran la mejora en este dataset y entorno, pero no implican
que ambas versiones sean numéricamente equivalentes para todas las entradas ni
que estos tiempos se puedan extrapolar sin medir otras cargas.
