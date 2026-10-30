# Resultados

> Las exploraciones de Numba y Rust para 2.3 se conservan por separado en
> [`experiments/`](experiments/README.md). Este documento mantiene las mediciones
> históricas de 2.2 y añade la comparación de las referencias 2.2 y 2.3.

Todas las campañas usan un contrato compacto común: `campaign.json` conserva
los metadatos y resúmenes, `measurements.jsonl` agrupa las repeticiones y
`validations.jsonl` guarda cada secuencia de índices una sola vez. Las campañas
anteriores a 2.3 se migraron mecánicamente y conservan la procedencia de sus
archivos originales.

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
uv run python -m benchmarks.run --counts 5 10 20 --runs 5 --timeout 180 \
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

## 2.1 frente a 2.2

### Qué cambió

La versión 2.1 ya había reducido el algoritmo a `O(n * d * k)`. La versión 2.2
no cambia ese crecimiento: reduce el coste del bucle que se repite para cada
nueva selección.

- Las desviaciones se conservan como un array de NumPy.
- `differences` y `normalized` se reservan una vez y se reutilizan con `out`.
- `normalized` usa el dtype resultante de combinar datos y desviaciones, para
  evitar truncar la división de entradas enteras.
- Las distancias se comparan al cuadrado mediante un helper con `einsum`, sin
  calcular una raíz cuadrada por fila.
- Si solo se solicita la semilla, no se reservan los buffers.

La raíz cuadrada es monótona, por lo que eliminarla no cambia el orden de las
distancias. Se mantienen la corrección angular, las distancias mínimas
incrementales, los desempates por menor índice y el dtype de salida. No cambia
la API pública, no se añaden dependencias y los números complejos quedan fuera
del alcance de esta subfase.

### Exploración de alternativas

Durante el desarrollo se midieron por separado tres variantes: buffers con
`linalg.norm`, distancias cuadradas con temporales por iteración y ambas
modificaciones combinadas. Los buffers por sí solos oscilaron entre una pequeña
mejora y una pequeña regresión; las distancias cuadradas aportaron la mayor
parte de la aceleración, alrededor de 1,26× a 1,53× en las cargas lineales.

La combinación quedó cerca de la variante de distancias cuadradas y mostró una
señal favorable de memoria en la carga de un millón de filas: unos 215 MiB de
RSS frente a 243 MiB. Es la implementación de 2.2 porque mantiene la
aceleración, evita dos asignaciones por iteración y sigue siendo legible con el
cálculo `einsum` encapsulado.

### Primera comparación: aislar los ejes de escalado

El **21 de octubre de 2026** se midieron 19 cargas para variar por separado
filas, columnas, selecciones, dtype y número de variables direccionales. Cada
carga se ejecutó siete veces en procesos independientes, alternando el orden de
las referencias. La pregunta de esta primera comparación era cómo cambiaba el
beneficio al mover un eje cada vez, no si las cargas representaban usos reales
completos.

```bash
uv run python -m benchmarks.scale --runs 7 --timeout 180 \
  --output benchmarks/results/v2_1-v2_2-scaling
```

[Datos brutos y resumen de escalado](results/v2_1-v2_2-scaling/)

La campaña conserva el hash de la primera implementación de 2.2. Esa referencia
todavía incluía un fallback para entradas complejas y mantenía las desviaciones
como lista, aunque las entradas medidas ya seguían la ruta de buffers y
distancias cuadradas. Los resultados conservan esa procedencia.

La tabla resume las medianas de selección por eje. Los tiempos absolutos de
estas cargas son inferiores a un segundo; esta campaña sirve para observar el
escalado, no para estimar ahorro práctico.

| Eje | Rango medido | Aceleración de 2.2 | Lectura |
| :--- | :--- | ---: | :--- |
| `datawave` | `k=5–100` | 1,06–1,49× | Mejora pequeña en tiempo absoluto. |
| filas | `n=10.000–1.000.000` | 1,51–1,65× | Mantiene el crecimiento lineal. |
| columnas | `d=1–30` | 1,46–1,62× | La reducción sigue siendo eficaz al crecer `d`. |
| selecciones | `k=5–500` | 1,47–1,83× | El ahorro se acumula al crecer `k`. |
| dtype | `float32–float64` | 1,73–1,87× | Mejora en ambas precisiones medidas. |
| direccionales | `0–3` variables | 1,09–1,74× | El módulo angular reduce el beneficio. |

El crecimiento fue aproximadamente lineal al variar filas y selecciones. La
mejora disminuyó al aumentar el trabajo angular, porque el módulo de los ángulos
pasó a dominar una parte mayor del bucle. Esta fase confirmó el comportamiento
por ejes, pero no combinaba muchas filas con muchas selecciones ni demostraba un
ahorro absoluto relevante.

### Segunda comparación: cargas documentadas

La segunda fase respondió a esa limitación. En vez de aislar un eje, combinó
filas y selecciones con tamaños encontrados en aplicaciones publicadas de MDA.
Se utilizó la implementación definitiva de 2.2.

La confirmación definitiva se ejecutó el **27 de octubre de 2026** en Linux
sobre WSL2 ARM64, con 12 CPU lógicas, Python 3.12.3, NumPy 1.26.0 y SciPy
1.15.1. Cada carga se ejecutó cinco veces en procesos independientes, alternando
el orden de 2.1 y 2.2.

```bash
uv run python -m benchmarks.scale --suite grounded --runs 5 --timeout 900 \
  --output benchmarks/results/v2_1-v2_2-grounded
```

[Datos brutos, entorno y resumen](results/v2_1-v2_2-grounded/)

Además de `datawave`, se construyeron datos sintéticos deterministas con tres
variables y semilla 20261020. Sus tamaños reproducen escalas publicadas para
MDA: 87.664 filas y 600 selecciones; 121.968 filas y entre 100 y 5.000
selecciones; un millón de filas y 1.250 selecciones; y la muestra de 10.000
filas y 1.000 selecciones utilizada antes de MDA en otro trabajo. Son cargas
comparables por tamaño, no reproducciones de esos datasets ni de sus variantes
del algoritmo.

La tabla muestra las medianas de selección aislada. Solo `datawave` utiliza un
dataset real del repositorio; las demás filas reproducen tamaños documentados
con datos sintéticos. El RSS es el máximo del proceso completo.

| Referencia de escala | Filas | Selecciones | 2.1 (s) | 2.2 (s) | Ahorro | Aceleración | Cambio RSS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `datawave` | 140.256 | 20 | 0,078 | 0,061 | 0,017 s | 1,28× | −4,4 % |
| `datawave` | 140.256 | 100 | 0,360 | 0,296 | 0,064 s | 1,21× | −2,8 % |
| Franklin et al. | 87.664 | 600 | 0,916 | 0,625 | 0,291 s | 1,46× | −3,9 % |
| Castro et al. | 121.968 | 100 | 0,218 | 0,139 | 0,079 s | 1,56× | −1,5 % |
| Castro et al. | 121.968 | 1.000 | 2,137 | 1,476 | 0,661 s | 1,45× | −6,7 % |
| Castro et al. | 121.968 | 5.000 | 11,220 | 7,418 | 3,802 s | 1,51× | −1,3 % |
| NC State | 1.000.000 | 1.250 | 22,812 | 15,167 | 7,645 s | 1,50× | −12,7 % |
| TU Delft | 10.000 | 1.000 | 0,156 | 0,106 | 0,050 s | 1,47× | +1,6 % |
| Control direccional | 87.664 | 600 | 1,329 | 1,215 | 0,115 s | 1,09× | −4,9 % |

La llamada completa mejoró en las nueve cargas, entre 1,10× y 1,56×. El ahorro
es pequeño en `datawave` con 20 selecciones, pero alcanza 3,80 s con 121.968
filas y 5.000 selecciones, y 7,64 s con un millón de filas y 1.250 selecciones.
La carga direccional mejora solo 1,09× porque la corrección angular no cambia.

La preparación quedó entre 0,0003 y 0,0174 segundos, por lo que la selección
domina las cargas grandes. El RSS mediano bajó en ocho cargas; en la mayor pasó
de 250,1 a 218,3 MiB. El aumento del 1,6 % con 10.000 filas es pequeño frente
al coste del proceso y no se atribuye al núcleo de forma aislada.

### Corrección y decisión

2.1 y 2.2 produjeron exactamente los mismos índices y en el mismo orden para
los nueve casos numéricos, las 19 cargas de escalado y las nueve cargas de
confirmación. Se cubrieron empates, duplicados, casi empates, `float32`,
`float64`, variables direccionales, hasta 30 columnas, un millón de filas y
5.000 selecciones. No se aplicaron tolerancias a los índices.

La equivalencia matemática entre una distancia y su cuadrado no garantiza por
sí sola los mismos resultados en coma flotante. Por eso se compararon índices,
incluidos los casos de empate y casi empate, en vez de aceptar únicamente
distancias próximas.

La versión 2.2 utiliza la implementación combinada. Mejora todas las cargas de
confirmación, aporta ahorros absolutos relevantes en las mayores, no presenta
un aumento material de memoria y sigue siendo explicable con operaciones NumPy
pequeñas. La velocidad procede principalmente de las distancias cuadradas; los
buffers se mantienen para evitar asignaciones y por su comportamiento de
memoria. Numba y cualquier cambio de API quedan fuera de esta subfase.

Los tiempos pertenecen a este equipo y a datos mayoritariamente sintéticos. No
deben extrapolarse directamente a otros entornos, distribuciones de datos o
implementaciones de MDA.

## Referencia NumPy 2.2 frente a Numba 2.3

### Objetivo y protocolo

Los experimentos previos sirvieron para elegir la forma del núcleo y estudiar
el número de hilos. Esta campaña no vuelve a comparar prototipos: compara las
referencias independientes de los motores NumPy 2.2 y Numba 2.3. Antes de medir,
ambas referencias se contrastan con la API pública
`mdapy.max_diss_alg(..., engine=...)` del checkout actual.

Se eligieron tres cargas para representar comportamientos distintos:

- `columns-d1`: 100.000 filas, una columna y 20 selecciones; una carga pequeña
  para la que el coste fijo de Numba debe quedar visible.
- `grounded-datawave-k100`: las 140.256 filas y tres variables de `datawave`,
  con 100 selecciones.
- `grounded-castro-k5000`: 121.968 filas sintéticas, tres variables y 5.000
  selecciones; una carga pesada basada en una escala publicada.

Cada combinación de carga y motor se ejecutó cinco veces en procesos aislados,
alternando el orden de los motores. En cada proceso se midió una primera llamada
y una segunda llamada caliente con los mismos datos. La campaña utilizó una caché
temporal de Numba compartida entre procesos; antes de las mediciones se ejecutaron
sondas separadas con caché vacía y caliente para no confundir compilación, carga
de código compilado y ejecución repetida.

La medición se realizó el **30 de octubre de 2026** en Linux sobre WSL2 ARM64,
con 12 CPU lógicas, Python 3.12.3, NumPy 1.26.0, SciPy 1.15.1, Numba 0.68.1 y
llvmlite 0.50.0. Se fijaron externamente `NUMBA_NUM_THREADS=12` y
`NUMBA_CPU_NAME=generic`; la primera variable hace explícito el número de hilos
OpenMP y la segunda mantiene la compatibilidad con LLVM en esta máquina. La
biblioteca no modifica ninguna de ellas.

```bash
NUMBA_CPU_NAME=generic NUMBA_NUM_THREADS=12 \
  uv run python -m benchmarks.numba --runs 5 \
  --workloads columns-d1 grounded-datawave-k100 grounded-castro-k5000 \
  --output benchmarks/results/v2_2-v2_3
```

[Resultados compactos, entorno y muestras](results/v2_2-v2_3/)

### Coste frío de Numba

La primera especialización incluye la compilación JIT. Como todos los cálculos
de 2.3 usan `float64`, los distintos dtypes de entrada comparten el mismo núcleo
compilado. La sonda utilizó 100.000 filas, tres columnas y 5 selecciones.
`Primera sin caché` parte de un directorio vacío; `primera con caché` arranca
otro proceso y carga el código ya compilado; `caliente` es la segunda llamada
dentro de ese proceso.

| Dtype | Primera sin caché (s) | Primera con caché (s) | Caliente (s) | Proceso sin caché (s) | Proceso con caché (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `float64` | 1,038 | 0,218 | 0,010 | 1,579 | 0,658 |

La caché reduce mucho el coste entre procesos, pero no lo elimina: todavía hay
que importar Numba y cargar la especialización. Por eso el motor no resulta
adecuado para una única carga pequeña aunque el núcleo compilado sea rápido.

### Rendimiento de las referencias

Los tiempos son medianas. `Primera` carga la referencia, prepara los datos y
selecciona con la caché de disco ya disponible; `caliente` repite preparación y
selección con el módulo y la especialización cargados. La aceleración divide el
tiempo caliente de NumPy entre el de Numba.

| Carga | NumPy primera (s) | Numba primera (s) | NumPy caliente (s) | Numba caliente (s) | Aceleración caliente |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `columns-d1` | 0,0061 | 0,2263 | 0,0047 | 0,0131 | 0,36× |
| `grounded-datawave-k100` | 0,3261 | 0,2977 | 0,2690 | 0,0606 | 4,44× |
| `grounded-castro-k5000` | 7,6558 | 1,4265 | 7,5952 | 0,9443 | 8,04× |

En la carga de una columna, Numba fue unas 37 veces más lento en la primera
llamada y 2,78 veces más lento en caliente. En `datawave`, la primera llamada
quedó próxima y la repetición ahorró 0,208 segundos. En la carga de 5.000
selecciones, Numba fue 5,37 veces más rápido en la primera llamada y 8,04 veces
en caliente, con ahorros de 6,23 y 6,65 segundos respectivamente.

El tiempo total mediano del proceso fue 0,416 frente a 0,740 segundos en la
carga pequeña, 0,990 frente a 0,831 segundos en `datawave` y 15,625 frente a
2,862 segundos en la carga pesada, siempre en orden NumPy frente a Numba. Esto
incluye arranque, imports, datos y ambas llamadas.

La diferencia de RSS máximo mediano quedó por debajo del 0,6 % en las tres
cargas. Esto no demuestra una reducción de memoria frente a NumPy: el RSS
corresponde al proceso completo y no a una medición aislada de los buffers del
núcleo.

### Corrección y decisión

Las referencias NumPy y Numba produjeron exactamente los mismos índices, en el
mismo orden, y el mismo hash de los datos seleccionados en las 30 ejecuciones
medidas. También se compararon las dos referencias y los dos motores públicos
con los nueve casos de [`cases.json`](cases.json), tanto en `float32` como en
`float64`. Se cubrieron empates, duplicados, casi empates y variables
direccionales; no se aplicaron tolerancias a los índices.

Los resultados respaldan mantener NumPy como motor predeterminado: evita una
dependencia y un coste fijo que perjudican claramente las cargas pequeñas. Numba
aporta valor como optimización opcional y explícita para cargas grandes o llamadas
repetidas. La API no expone el número de hilos en esta primera versión; Numba
respeta su configuración externa. En esta máquina, usar los 12 hilos fijados no
siempre fue la configuración experimental más rápida, pero mantiene el contrato
sencillo y conserva una mejora amplia en la carga pesada.

Esta comparación se limita a un entorno ARM64, tres cargas y dos dtypes. Los
tiempos no establecen un umbral automático entre motores ni deben extrapolarse a
otros equipos. La decisión queda en manos del usuario mediante `engine="numpy"`
o `engine="numba"`, sin selección automática ni fallback silencioso.
