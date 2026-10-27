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

```bash
uv run benchmarks/run.py --counts 5 10 20 --runs 5 --timeout 180 \
  --output benchmarks/results/v2_0-v2_1
```

[Datos brutos de la campaña](results/v2_0-v2_1/)

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

## Escalado de 2.1 y candidato Python/NumPy para 2.2

Medición del **21 de octubre de 2026** en el mismo entorno ARM64 descrito
arriba, con Python 3.12.3, NumPy 1.26.0 y SciPy 1.15.1. Se comparó la referencia
publicada `v2.1.0` con el candidato de la rama `perf-2.2-python-numpy`.

La complejidad asintótica no cambia: ambas versiones realizan
`O(n * d * k)` trabajo y utilizan `O(n * d)` memoria auxiliar. El candidato
reutiliza las matrices de diferencias y normalización entre iteraciones y, para
datos reales, compara distancias euclídeas al cuadrado mediante `einsum`. La
raíz cuadrada es monótona y no cambia el orden. Las entradas complejas conservan
la reducción con `np.linalg.norm`.

Se midieron datos sintéticos deterministas con semilla 20261020, además de
`datawave`. Cada carga se ejecutó siete veces en procesos independientes,
alternando el orden de las referencias. La preparación, selección, llamada
completa y memoria RSS máxima se registraron por separado.

```bash
uv run benchmarks/scale.py --runs 7 --timeout 180 \
  --output benchmarks/results/v2_1-v2_2-scaling
```

[Datos brutos y resumen de la campaña](results/v2_1-v2_2-scaling/)

### Corrección

Las dos referencias produjeron exactamente los mismos índices, en el mismo
orden, para los nueve casos numéricos y las 19 cargas medidas. Esto incluye
empates, duplicados, casi empates en `float32` y `float64`, columnas
direccionales, hasta un millón de filas, 30 columnas y 500 selecciones. No se
aplicaron tolerancias a los índices.

### Tiempos

La tabla muestra medianas del tiempo de selección aislado. `Aceleración` es el
cociente entre 2.1 y el candidato; un valor mayor que uno favorece a 2.2.

| Eje | Carga | 2.1 (s) | Candidato 2.2 (s) | Aceleración |
| :--- | :---- | ------: | ------------------: | -----------: |
| real | `datawave`, `k=5` | 0,028060 | 0,018819 | 1,49× |
| real | `datawave`, `k=20` | 0,072310 | 0,068450 | 1,06× |
| real | `datawave`, `k=100` | 0,365442 | 0,322873 | 1,13× |
| filas | `n=10.000` | 0,003594 | 0,002324 | 1,55× |
| filas | `n=100.000` | 0,040259 | 0,024412 | 1,65× |
| filas | `n=1.000.000` | 0,381650 | 0,253087 | 1,51× |
| columnas | `d=1` | 0,006963 | 0,004778 | 1,46× |
| columnas | `d=3` | 0,036088 | 0,023501 | 1,54× |
| columnas | `d=10` | 0,062780 | 0,038647 | 1,62× |
| columnas | `d=30` | 0,145054 | 0,089312 | 1,62× |
| selecciones | `k=5` | 0,013024 | 0,007108 | 1,83× |
| selecciones | `k=20` | 0,039170 | 0,023403 | 1,67× |
| selecciones | `k=100` | 0,177555 | 0,113294 | 1,57× |
| selecciones | `k=500` | 0,883417 | 0,599381 | 1,47× |
| dtype | `float32` | 0,051147 | 0,029546 | 1,73× |
| dtype | `float64` | 0,040141 | 0,021446 | 1,87× |
| dirección | ninguna direccional | 0,038376 | 0,022066 | 1,74× |
| dirección | una direccional | 0,051195 | 0,043601 | 1,17× |
| dirección | todas direccionales | 0,086541 | 0,079580 | 1,09× |

El crecimiento observado es aproximadamente lineal al variar por separado las
filas y las selecciones, de acuerdo con `O(n * d * k)`. La mejora disminuye al
aumentar el trabajo angular: con todas las columnas direccionales, el cálculo
del módulo domina y la reducción de distancia aporta solo un 8,7 %.

La llamada completa mejoró en las 19 cargas, entre 1,05× y 1,87×. La mediana
del RSS máximo varió entre un 14,1 % menos y un 1,3 % más. Este RSS incluye el
intérprete, imports y datos, por lo que no aísla únicamente los buffers del
núcleo.

### Decisión

El candidato supera el criterio acordado: mejora claramente varias cargas
representativas, no presenta regresiones temporales, conserva los índices
exactos y no aumenta materialmente la memoria. La reutilización de buffers y
las distancias cuadradas se incorporan juntas como evolución Python/NumPy de
2.2. La acumulación por columnas estudiada durante la exploración no se adopta:
fue muy rápida con tres variables, pero aproximadamente 3,6 veces más lenta que
2.1 con 30 columnas. Numba queda fuera de esta subfase.
