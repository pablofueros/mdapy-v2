# Evaluación de Rust paralelo con Rayon para `mdapy` 2.3

## Alcance y conclusión

Esta campaña añade paralelismo por filas al candidato Rust secuencial mediante
Rayon. Conserva `rust-script` como puente experimental y no añade PyO3,
`maturin`, Cargo ni dependencias al paquete Python. NumPy, Numba secuencial,
Numba paralelo, Rust secuencial y Rust paralelo se miden de nuevo bajo el mismo
protocolo.

Rayon mejora Rust cuando cada sincronización contiene suficiente trabajo, pero
no alcanza al candidato Numba paralelo:

- reprodujo exactamente los mismos índices y su orden con 1, 2, 4, 8 y 12
  hilos, en los casos de referencia, las cinco cargas de ajuste, las nueve de
  confirmación y las 19 de escalado;
- cuatro hilos fueron la mejor configuración del entorno, con una mediana de
  1,69× frente a Rust secuencial en las cinco cargas de ajuste;
- en confirmación mejoró Rust secuencial en 9/9, con mediana 1,24×, y NumPy en
  9/9, con mediana 3,92×;
- Numba paralelo ganó las nueve cargas de confirmación y fue aproximadamente
  2,10× más rápido en la mediana;
- en escalado, Rust paralelo mejoró Rust secuencial en 9/19 y su mediana fue
  1,00×: el beneficio desaparece al incluir cargas pequeñas;
- frente a Numba paralelo solo ganó 3/19 casos de escalado, todos con poco
  trabajo, y quedó en 0,79× en la mediana.

La variante Rayon confirma que Rust puede paralelizar el núcleo sin cambiar su
resultado, pero esta estrategia introduce una barrera por selección. Con miles
de selecciones, el coste repetido limita el escalado y demasiados hilos pueden
ser mucho peores que el núcleo secuencial.

## Diseño y determinismo

En cada iteración, `par_iter_mut` distribuye entre hilos las filas de la matriz.
Cada tarea calcula la distancia de sus filas y actualiza posiciones distintas
del vector de distancias mínimas. No existen reducciones de coma flotante entre
hilos: la suma de columnas de una fila conserva el mismo orden serial que la
referencia.

Cuando termina la actualización paralela, todos los hilos se sincronizan. La
búsqueda del siguiente máximo permanece serial y recorre los índices en orden,
por lo que los empates exactos siguen favoreciendo al primero. La siguiente
selección no puede comenzar antes porque depende del índice elegido.

El pool Rayon se crea una vez por proceso y queda fuera de las mediciones del
núcleo. Cada selección reutiliza ese pool, pero todavía debe enviar el trabajo y
esperar una barrera. No se usa `unsafe` ni matemática rápida.

## Dependencia experimental

El script contiene un manifiesto Cargo embebido:

```toml
[dependencies]
rayon = ">=1.10, <2"
```

`rust-script` descargó y resolvió Rayon 1.12.0 y `rayon-core` 1.13.0 en su caché
de Cargo. Ningún manifiesto o lock de Rust forma parte del proyecto y no se
modificaron `pyproject.toml` ni `uv.lock`.

## Ajuste del número de hilos

Se midieron cinco cargas representativas con siete repeticiones para 1, 2, 4, 8
y 12 hilos. La tabla muestra aceleración frente al candidato Rust secuencial:

| Hilos | Mediana | Mínimo | Máximo | Cargas más rápidas |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0,62× | 0,61× | 0,88× | 0/5 |
| 2 | 1,07× | 0,85× | 1,49× | 3/5 |
| 4 | 1,69× | 1,25× | 2,49× | 5/5 |
| 8 | 1,37× | 0,47× | 1,70× | 3/5 |
| 12 | 0,31× | 0,14× | 0,81× | 0/5 |

Cuatro hilos son la única configuración que mejoró las cinco cargas. Un solo
hilo Rayon es más lento que el bucle Rust directo por su infraestructura. Con
Castro 5.000, ocho y doce hilos retrocedieron a 0,47× y 0,14×: coordinar 5.000
barreras domina cualquier paralelismo adicional.

## Confirmación con cuatro hilos

La tabla muestra medianas del núcleo de selección caliente:

| Carga | NumPy (s) | Numba sec. (s) | Numba par. (s) | Rust sec. (s) | Rust par. (s) | Rust par./sec. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `datawave`, k=20 | 0,0606 | 0,0162 | 0,0103 | 0,0166 | 0,0131 | 1,27× |
| `datawave`, k=100 | 0,3136 | 0,0824 | 0,0265 | 0,0826 | 0,0549 | 1,50× |
| Franklin, k=600 | 0,6278 | 0,1929 | 0,0689 | 0,1944 | 0,1691 | 1,15× |
| Castro, k=100 | 0,1446 | 0,0448 | 0,0157 | 0,0464 | 0,0444 | 1,04× |
| Castro, k=1.000 | 1,4712 | 0,4580 | 0,1554 | 0,4631 | 0,3755 | 1,23× |
| Castro, k=5.000 | 7,4299 | 2,3950 | 0,8798 | 2,3565 | 1,8945 | 1,24× |
| NC State, k=1.250 | 15,2300 | 4,6926 | 1,8865 | 4,6633 | 2,4132 | 1,93× |
| TU Delft, k=1.000 | 0,1160 | 0,0380 | 0,0161 | 0,0374 | 0,0339 | 1,11× |
| Direccional, k=600 | 1,1941 | 0,4761 | 0,1219 | 0,4024 | 0,2246 | 1,79× |

Rust paralelo mejoró NumPy entre 3,25× y 6,31×. Frente a Numba secuencial
ganó también 9/9, con mediana 1,24×. Sin embargo, alcanzó entre 0,35× y 0,79×
del rendimiento de Numba paralelo.

## Matriz de escalado

| Referencia | Mínimo | Mediana | Máximo | Victorias de Rust paralelo |
| :--- | ---: | ---: | ---: | ---: |
| NumPy 2.2 | 0,37× | 3,09× | 7,21× | 18/19 |
| Numba secuencial | 0,29× | 0,84× | 2,46× | 9/19 |
| Numba paralelo | 0,38× | 0,79× | 5,17× | 3/19 |
| Rust secuencial | 0,60× | 1,00× | 2,41× | 9/19 |

La única regresión frente a NumPy volvió a ser una columna, donde Rust
paralelo quedó en 0,37×. Las victorias frente a Numba paralelo fueron
`datawave` con cinco selecciones, 10.000 filas y el control sin columnas
direccionales; son casos donde el propio Numba paralelo sufre costes fijos.

Rust paralelo fue favorable respecto a Rust secuencial con más trabajo por
iteración: un millón de filas, diez o treinta columnas, cien o quinientas
selecciones y las cargas direccionales más costosas. No fue favorable de forma
general con 100.000 filas y tres columnas.

## Compilación y limitaciones

Con las dependencias de Cargo ya disponibles, compilar una copia fría del
script paralelo costó entre 0,93 y 1,08 segundos. Invocarla desde la caché costó
entre 2,8 y 5,1 ms en la sonda pequeña. La primera descarga y compilación de
Rayon depende del estado de la caché y de la red, por lo que no se presenta como
una medida reproducible de instalación.

La CLI continúa usando un archivo temporal y un proceso separado. Estos costes
no representan una futura extensión PyO3. Tampoco se comparan RSS, tamaño de
wheel ni complejidad de distribución multiplataforma.

Una implementación Rust más compleja podría mantener a los trabajadores dentro
de un bucle persistente o paralelizar parte de la reducción. Eso exigiría nuevas
reglas de sincronización y desempate y queda fuera de este experimento educativo.

## Decisión experimental

Rayon demuestra una mejora real para cargas pesadas, pero la implementación
actual no compite con Numba paralelo. Rust conserva posibles ventajas de AOT y
control de distribución, pero convertirlas en un motor Python exigiría PyO3,
wheels por plataforma y más mantenimiento sin una ventaja de rendimiento
demostrada.

Esta campaña cierra la exploración de rendimiento previa al análisis de 2.3. La
decisión posterior debe valorar conjuntamente rendimiento, primera llamada,
instalación, compatibilidad, mantenimiento y objetivos educativos. No se propone
todavía ningún cambio en la API.

## Reproducción

```bash
NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.rust.evaluate_parallel --check \
  --thread-counts 1 2 4 8 12

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.rust.evaluate_parallel \
  --mode tuning --thread-counts 1 2 4 8 12 --runs 7 \
  --output benchmarks/experiments/rust/results/parallel/tuning

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.rust.evaluate_parallel \
  --mode comparison --suite grounded --threads 4 --numba-threads 4 --runs 7 \
  --output benchmarks/experiments/rust/results/parallel/grounded-t4

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.rust.evaluate_parallel \
  --mode comparison --suite scaling --threads 4 --numba-threads 4 --runs 7 \
  --output benchmarks/experiments/rust/results/parallel/scaling-t4
```

Cada directorio usa el [formato compacto de experimentos](../README.md#resultados-compactos)
para conservar metadatos, hashes de los cinco candidatos, todas las muestras,
índices exactos, sondas de compilación y el resumen sin duplicar información.
