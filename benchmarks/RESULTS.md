# Resultados

> Las exploraciones de Numba y Rust para 2.3 se conservan por separado en
> [`experiments/`](experiments/README.md). Este documento mantiene las mediciones
> históricas de 2.2 y añade la confirmación de la implementación pública de 2.3.

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
uv run python benchmarks/scale.py --runs 7 --timeout 180 \
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
uv run python benchmarks/scale.py --suite grounded --runs 5 --timeout 900 \
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

## Confirmación del motor Numba de 2.3

### Objetivo y protocolo

Los experimentos previos sirvieron para elegir la forma del núcleo y estudiar
el número de hilos. Esta campaña no vuelve a comparar prototipos: confirma la
implementación candidata a 2.3 mediante la API pública
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

La medición se realizó el **29 de octubre de 2026** en Linux sobre WSL2 ARM64,
con 12 CPU lógicas, Python 3.12.3, NumPy 1.26.0, SciPy 1.15.1, Numba 0.68.1 y
llvmlite 0.50.0. Numba utilizó sus 12 hilos predeterminados mediante OpenMP. En
esta máquina se fijó externamente `NUMBA_CPU_NAME=generic` por compatibilidad con
LLVM; la biblioteca no modifica esa variable ni el número de hilos.

```bash
NUMBA_CPU_NAME=generic uv run python -m benchmarks.numba --runs 5 \
  --workloads columns-d1 grounded-datawave-k100 grounded-castro-k5000 \
  --output benchmarks/results/v2_3-numba-public
```

[Resultados compactos, entorno y muestras](results/v2_3-numba-public/)

### Coste frío de Numba

La primera especialización de cada dtype incluye la compilación JIT. Las sondas
utilizaron 100.000 filas y tres columnas: 5 selecciones para `float64` y 20 para
`float32`. `Primera sin caché` parte de un directorio vacío; `primera con caché`
arranca otro proceso y carga el código ya compilado; `caliente` es la segunda
llamada dentro de ese proceso.

| Dtype | Primera sin caché (s) | Primera con caché (s) | Caliente (s) | Proceso sin caché (s) | Proceso con caché (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `float64` | 0,907 | 0,185 | 0,005 | 1,444 | 0,627 |
| `float32` | 0,789 | 0,237 | 0,010 | 1,292 | 0,726 |

La caché reduce mucho el coste entre procesos, pero no lo elimina: todavía hay
que importar Numba y cargar la especialización. Por eso el motor no resulta
adecuado para una única carga pequeña aunque el núcleo compilado sea rápido.

### Rendimiento mediante la API pública

Los tiempos son medianas. `Primera` es la primera llamada pública del proceso
con la caché de disco ya disponible; `caliente` es la repetición inmediata. La
aceleración divide el tiempo caliente de NumPy entre el de Numba.

| Carga | NumPy primera (s) | Numba primera (s) | NumPy caliente (s) | Numba caliente (s) | Aceleración caliente |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `columns-d1` | 0,0049 | 0,2104 | 0,0034 | 0,0109 | 0,32× |
| `grounded-datawave-k100` | 0,2834 | 0,2774 | 0,2321 | 0,0401 | 5,79× |
| `grounded-castro-k5000` | 6,4500 | 1,7248 | 6,3452 | 0,8482 | 7,48× |

En la carga de una columna, Numba fue unas 43 veces más lento en la primera
llamada y 3,17 veces más lento en caliente. En `datawave`, la primera llamada
quedó prácticamente empatada y la repetición ahorró 0,192 segundos. En la carga
de 5.000 selecciones, Numba ya fue 3,74 veces más rápido en la primera llamada
y 7,48 veces en caliente, con ahorros de 4,73 y 5,50 segundos respectivamente.

El tiempo total mediano del proceso fue 0,372 frente a 0,669 segundos en la
carga pequeña, 0,855 frente a 0,743 segundos en `datawave` y 13,130 frente a
3,167 segundos en la carga pesada, siempre en orden NumPy frente a Numba. Esto
incluye arranque, imports, datos y ambas llamadas.

El RSS máximo mediano de Numba aumentó un 1,2 %, un 5,0 % y un 5,2 % en las tres
cargas, respectivamente. Es un coste moderado del runtime y del código compilado,
no una reducción de memoria frente a NumPy. El RSS corresponde al proceso completo
y no a una medición aislada de los buffers del núcleo.

### Corrección y decisión

NumPy y Numba produjeron exactamente los mismos índices, en el mismo orden, y el
mismo hash de los datos seleccionados en las 30 ejecuciones medidas. También se
compararon mediante la API pública los nueve casos de referencia de
[`cases.json`](cases.json) tanto en `float32` como en `float64`. Se cubrieron
empates, duplicados, casi empates y variables direccionales; no se aplicaron
tolerancias a los índices.

Los resultados respaldan mantener NumPy como motor predeterminado: evita una
dependencia y un coste fijo que perjudican claramente las cargas pequeñas. Numba
aporta valor como optimización opcional y explícita para cargas grandes o llamadas
repetidas. La API no expone el número de hilos en esta primera versión; Numba
respeta su configuración externa. En esta máquina, usar los 12 hilos
predeterminados no siempre fue la configuración experimental más rápida, pero
mantiene el contrato sencillo y conserva una mejora amplia en la carga pesada.

Esta confirmación se limita a un entorno ARM64, tres cargas y dos dtypes. Los
tiempos no establecen un umbral automático entre motores ni deben extrapolarse a
otros equipos. La decisión queda en manos del usuario mediante `engine="numpy"`
o `engine="numba"`, sin selección automática ni fallback silencioso.
