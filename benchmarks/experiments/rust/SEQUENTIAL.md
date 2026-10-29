# Evaluación de Rust secuencial para `mdapy` 2.3

Esta página conserva la campaña secuencial como referencia independiente. La
[evaluación posterior con Rayon](PARALLEL.md) estudia la actualización de
distancias por filas con varios hilos.

## Alcance y conclusión

Este experimento estudia si un núcleo escrito en Rust merece una evaluación más
profunda. Utiliza `rust-script` como herramienta externa y no añade PyO3,
`maturin`, Cargo, Rayon ni dependencias al proyecto. Tampoco cambia la API
pública ni propone todavía una forma de distribución.

El resultado secuencial es prometedor, pero no justifica por sí solo sustituir
Numba:

- reprodujo exactamente los mismos índices y su orden para todos los casos de
  referencia, las nueve cargas de confirmación y las 19 de escalado;
- en confirmación mejoró NumPy entre 2,84× y 3,81×, con mediana 3,21×;
- en esas nueve cargas quedó prácticamente empatado con Numba secuencial:
  mediana 1,02× y seis victorias para Rust;
- en escalado mejoró NumPy en 18 de 19 cargas, con mediana 2,84×, pero quedó
  alrededor de un 10 % por detrás de Numba secuencial en la mediana;
- Numba paralelo siguió siendo claramente más rápido en todas las cargas de
  confirmación. Rust solo lo superó en seis casos de escalado pequeños donde el
  coste de coordinar hilos penaliza al candidato paralelo.

Rust secuencial confirma que una implementación AOT puede competir con el
núcleo Numba secuencial. Para comparar el mejor rendimiento potencial de ambas
tecnologías haría falta otra subfase con paralelismo en Rust. Esa subfase queda
fuera del alcance actual.

## Diseño

Python conserva la responsabilidad de cargar o generar las columnas y preparar
las desviaciones según la referencia 2.2. El evaluador escribe temporalmente
una representación binaria versionada con:

- dtype `float32` o `float64`;
- dimensiones, semilla y número de selecciones;
- máscara de columnas direccionales;
- desviaciones en `float64`;
- matriz contigua en orden de filas.

`rust_candidate.rs` lee esa entrada y ejecuta un núcleo fusionado equivalente
al candidato Numba secuencial. Mantiene seriales tanto la suma por fila como la
búsqueda del siguiente máximo. No usa paralelismo, `unsafe` ni transformaciones
de matemática rápida. Los slices e iteradores permiten al compilador eliminar
comprobaciones de límites redundantes sin cambiar el orden aritmético.

Cada proceso Rust ejecuta dos selecciones sobre la misma entrada. Los tiempos
internos se miden con `Instant`; por tanto, el tiempo del núcleo excluye el
arranque de `rust-script`, la lectura del archivo y la serialización desde
Python. Esos costes se registran también, pero por separado.

## Contrato numérico

La validación exige igualdad exacta de los índices y de su orden. Cubre:

- `float32` y `float64`;
- semillas distintas de cero;
- empates, duplicados y casi empates;
- una y varias columnas direccionales;
- ángulos negativos y superiores a una vuelta;
- hasta 30 columnas, un millón de filas y 5.000 selecciones.

La resta de datos `float32` se realiza antes de convertir su resultado a
`float64`, igual que en el candidato Numba. Para los ángulos, `rem_euclid`
reproduce el módulo positivo utilizado por NumPy y Numba. El desempate recorre
las filas en orden y conserva el primer índice con la distancia máxima.

## Entorno y protocolo

Las mediciones se ejecutaron el 29 de octubre de 2026 con:

- Python 3.12.3 y NumPy 1.26.0;
- Numba 0.68.1 y llvmlite 0.50.0;
- `rust-script` 0.36.0;
- Rust y Cargo 1.90.0, LLVM 20.1.8;
- Linux ARM64 bajo WSL2, 12 CPU;
- `NUMBA_CPU_NAME=generic` y cuatro hilos para Numba paralelo;
- semilla sintética `20261020`;
- siete repeticiones por algoritmo y carga, alternando su orden.

Se compararon NumPy 2.2, Numba secuencial, Numba paralelo y Rust secuencial. Los
tres primeros se volvieron a medir; no se reutilizaron los tiempos de campañas
anteriores. Solo `datawave` es un dataset real incluido en el repositorio. Las
otras cargas reproducen tamaños publicados mediante datos sintéticos.

## Resultados de confirmación

La tabla muestra medianas del núcleo de selección caliente. Un factor Rust/Numba
superior a uno significa que Rust fue más rápido.

| Carga | NumPy (s) | Numba sec. (s) | Numba par. (s) | Rust (s) | Rust/NumPy | Rust/Numba sec. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `datawave`, k=20 | 0,0630 | 0,0175 | 0,0079 | 0,0166 | 3,79× | 1,05× |
| `datawave`, k=100 | 0,3370 | 0,0916 | 0,0278 | 0,0884 | 3,81× | 1,04× |
| Franklin, k=600 | 0,6237 | 0,2035 | 0,0686 | 0,2035 | 3,06× | 1,00× |
| Castro, k=100 | 0,1539 | 0,0447 | 0,0160 | 0,0475 | 3,24× | 0,94× |
| Castro, k=1.000 | 1,4722 | 0,4639 | 0,1605 | 0,4535 | 3,25× | 1,02× |
| Castro, k=5.000 | 7,6351 | 2,4816 | 0,8956 | 2,4147 | 3,16× | 1,03× |
| NC State, k=1.250 | 15,6647 | 4,8141 | 2,0607 | 4,8862 | 3,21× | 0,99× |
| TU Delft, k=1.000 | 0,1207 | 0,0375 | 0,0170 | 0,0391 | 3,09× | 0,96× |
| Direccional, k=600 | 1,2031 | 0,4739 | 0,1273 | 0,4234 | 2,84× | 1,12× |

Numba paralelo fue entre 2,11× y 3,33× más rápido que Rust en estas cargas.
La ventaja de Rust frente a Numba secuencial fue pequeña y no uniforme; ambos
núcleos deben considerarse aproximadamente equivalentes en esta campaña.

## Matriz de escalado

| Referencia | Mínimo | Mediana | Máximo | Victorias de Rust |
| :--- | ---: | ---: | ---: | ---: |
| NumPy 2.2 | 0,85× | 2,84× | 4,08× | 18/19 |
| Numba secuencial | 0,60× | 0,91× | 1,26× | 7/19 |
| Numba paralelo | 0,31× | 0,81× | 8,47× | 6/19 |

La única regresión frente a NumPy apareció con una columna: Rust alcanzó
0,85×. Frente a Numba secuencial, Rust destacó con diez o treinta columnas y
con todas las columnas direccionales; quedó por detrás especialmente con una
columna y en las cargas aisladas de `float32` y `float64`.

Las seis victorias frente a Numba paralelo no corresponden a trabajo pesado:
son cargas con pocas filas, columnas o selecciones en las que la coordinación
de hilos cuesta más que el cálculo. En las nueve cargas de confirmación, donde
se usaría realmente un motor acelerado, Numba paralelo ganó siempre.

## Compilación e interfaz CLI

Una compilación fría aislada de `rust-script` tardó entre 0,50 y 0,56 segundos.
Invocar de nuevo el ejecutable cacheado sobre el caso de sonda tardó alrededor
de 3 ms. A diferencia de Numba, Rust no compila especializaciones separadas por
dtype durante el uso: ambas variantes forman parte del mismo binario AOT.

Para estimar una sola llamada CLI se resta la segunda selección interna al
tiempo total del proceso. En las cargas de confirmación, esa estimación fue una
mediana de 1,06× el tiempo del núcleo; el coste fijo se amortiza bien. En la
matriz de escalado fue 1,71× y llegó a 5,40× en las cargas más pequeñas.

Estos tiempos no predicen directamente una extensión Python. PyO3 podría evitar
el proceso y el archivo temporal, pero también introducir copias o conversiones
distintas. La campaña solo permite valorar el núcleo y cuantificar la penalización
del puente experimental actual.

No se comparan RSS ni tamaño de instalación: el proceso Rust es hijo del
evaluador Python y el toolchain ya estaba instalado. Una propuesta de
integración tendría que medir por separado memoria, binario y wheels.

## Decisión experimental

Rust secuencial queda validado como candidato educativo y técnicamente viable:
es aproximadamente tres veces más rápido que NumPy en cargas representativas y
puede igualar a Numba secuencial. Sin embargo, no supera al candidato Numba
paralelo para el uso pesado previsto, y `rust-script` no resuelve la integración
ni la distribución de una biblioteca Python.

Por tanto:

1. no se propone todavía `engine="rust"`;
2. no se justifican aún PyO3, `maturin` ni cambios de empaquetado;
3. una futura comparación de rendimiento máximo debería estudiar Rust paralelo,
   probablemente con Rayon, en una subfase independiente;
4. solo si esa variante compite con Numba paralelo tendría sentido evaluar el
   coste real de integración y wheels multiplataforma.

## Reproducción

La validación y las campañas requieren `rust-script` y una instalación temporal
de Numba, pero no modifican las dependencias del proyecto:

```bash
NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.rust.evaluate --check

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.rust.evaluate \
  --suite grounded --runs 7 --threads 4 \
  --output benchmarks/experiments/rust/results/grounded

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.rust.evaluate \
  --suite scaling --runs 7 --threads 4 \
  --output benchmarks/experiments/rust/results/scaling
```

Los metadatos conservan plataforma, toolchains, semillas, cargas, repeticiones y
hashes de los cuatro candidatos. Los archivos JSONL contienen las muestras
crudas y `compilation.json` separa compilación fría y ejecución cacheada.
