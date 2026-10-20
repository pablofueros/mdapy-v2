# Análisis de rendimiento

Esta carpeta compara evoluciones publicadas del algoritmo de máxima
disimilitud. No sigue automáticamente el código de desarrollo: cada referencia
se añade de forma deliberada cuando existe una nueva evolución que merece ser
comparada.

La comparación actual cubre:

- `v2_0.py`: comportamiento numérico de `v2.0.0`, último parche de la serie
  2.0 disponible al crear el análisis;
- `v2_1.py`: comportamiento numérico de `v2.1.0`, último parche de la serie
  2.1 disponible al crear el análisis.

Git conserva revisiones anteriores. Si aparece un parche posterior de una
serie, se actualiza su única referencia en lugar de añadir otra copia.

## Contrato numérico

Cada entrada contiene columnas unidimensionales, finitas y de igual longitud.
Las columnas direccionales representan grados con periodo 360. Las desviaciones
lineales usan `numpy.std` con `ddof=0`; las direccionales usan
`scipy.stats.circstd` con `high=360` y `low=0`.

Para dos observaciones, la diferencia direccional se reduce al intervalo
`[-180, 180)`:

```text
delta = (x - y + 180) % 360 - 180
```

La distancia normalizada de 2.1 es:

```text
sqrt(sum((delta_j / deviation_j) ** 2))
```

La versión 2.0 divide además cada término por `sqrt(2)`. Ese factor global no
cambia el orden matemático, aunque forma parte de la referencia histórica.

La selección comienza en `seed_index`. En cada iteración elige la observación
cuya distancia a su punto seleccionado más cercano sea máxima. Las filas ya
seleccionadas quedan excluidas y un empate exacto se resuelve a favor del menor
índice de entrada.

Dos ejecuciones son equivalentes para un caso cuando producen exactamente los
mismos índices y en el mismo orden. No se aplica una tolerancia a los índices ni
se convierten distancias cercanas en empates. Las diferencias conocidas entre
versiones se declaran de forma explícita en `cases.json`.

## Diferencias entre 2.0 y 2.1

La versión 2.0 convierte las columnas a `float32`, calcula repetidamente la
distancia de cada candidato al subconjunto seleccionado y conserva una caché de
pares. La versión 2.1 mantiene el dtype común determinado por NumPy, calcula las
distancias por lotes y guarda solamente la distancia mínima actual de cada
candidato.

Para `n` observaciones y `k` selecciones, el trabajo de selección pasa de
aproximadamente `O(n * k**2)` a `O(n * k)`. La caché de pares de 2.0 también
crece con las comparaciones realizadas, mientras que 2.1 mantiene vectores de
tamaño `n`.

## Casos y medición

`cases.json` fija entradas pequeñas, semilla, número de selecciones e índices
esperados para cada referencia. Antes de medir, el runner exige que todos los
casos pasen.

La carga reproducible usa todas las filas de `examples/data/datawave.npz`, las
columnas `hs`, `tp` y `dir`, y empieza en la fila con mayor `tp`. La campaña
principal selecciona 5, 10 y 20 observaciones, con cinco repeticiones y un
límite de 180 segundos por proceso. Esta progresión representa cargas pequeñas
y medias y permite observar el escalado sin llevar la referencia 2.0 a valores
desproporcionados. Cada repetición se ejecuta en un proceso independiente. Se
registran:

- versiones del entorno y hashes del dataset y las referencias;
- tamaño, dtype, semilla y número de selecciones;
- tiempo de carga, preparación, selección y proceso completo;
- memoria RSS máxima del proceso;
- índices elegidos en orden.

Ejecutar la validación rápida:

```bash
uv run benchmarks/run.py --check
```

Ejecutar la campaña predeterminada, con cinco procesos por carga:

```bash
uv run benchmarks/run.py
```

Para una comprobación corta, fuera de la campaña principal:

```bash
uv run benchmarks/run.py --counts 5 --runs 2 --timeout 60
```

Los datos brutos se escriben en `benchmarks/results/`, ignorado por Git. Cada
campaña debe usar un directorio vacío. Las conclusiones revisadas se documentan
en `RESULTS.md`.
