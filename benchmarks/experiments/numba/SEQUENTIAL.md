# Evaluación de Numba secuencial para `mdapy` 2.3

Esta página conserva la campaña secuencial como referencia independiente. La
[evaluación posterior del paralelismo](PARALLEL.md) compara ese mismo candidato
con un kernel Numba que distribuye entre hilos la actualización de distancias.

## Alcance y conclusión

Esta evaluación compara el núcleo NumPy de 2.2 con un candidato secuencial
compilado con Numba. No cambia la API pública ni añade dependencias al paquete.

El candidato supera el criterio acordado de corrección y rendimiento:

- produjo exactamente los mismos índices, en el mismo orden, para los casos de
  referencia, las 19 cargas de escalado y las nueve cargas de confirmación;
- mejoró las 19 cargas de escalado entre 1,37× y 4,77× en caliente;
- mejoró las cinco cargas principales entre 2,47× y 3,32×, con ahorros entre
  0,35 y 8,90 segundos;
- las cinco cargas principales superaron 1,5× y ahorraron más de 0,25 segundos,
  sin regresiones superiores al 10 %.

La incorporación a la biblioteca queda condicionada, no aprobada todavía. La
generación de código nativo abortó dentro de LLVM en este equipo ARM64 bajo
WSL2. `NUMBA_CPU_NAME=generic` evitó el fallo y se utilizó en todas las
mediciones. El mismo error de backend AArch64 continúa documentado en la
[incidencia abierta #10388 de Numba](https://github.com/numba/numba/issues/10388).
Antes de diseñar la API opcional se debe reproducir la evaluación en las
plataformas soportadas y resolver cómo tratar esta incompatibilidad sin alterar
globalmente la configuración de Numba de los usuarios.

## Contrato numérico

La preparación del candidato reproduce la referencia 2.2. El núcleo compilado
fusiona en bucles explícitos el cálculo de distancias, la actualización de la
distancia mínima y la elección siguiente. Se ejecuta en modo estricto, sin
`fastmath` y sin `parallel=True`.

Las comprobaciones exigen igualdad exacta de los índices y de su orden. Cubren:

- `float32` y `float64`;
- empates, duplicados y casi empates;
- semillas distintas de cero;
- una o varias columnas direccionales y ángulos con varias vueltas;
- hasta 30 columnas, un millón de filas y 5.000 selecciones.

No se aplicaron tolerancias a los índices ni se cambiaron expectativas para el
candidato.

## Entorno y protocolo

Las dos campañas se ejecutaron el 28 de octubre de 2026 con:

- Python 3.12.3;
- NumPy 1.26.0 y SciPy 1.15.1;
- Numba 0.68.1 y llvmlite 0.50.0;
- Linux ARM64 bajo WSL2, 12 CPU;
- `NUMBA_CPU_NAME=generic`;
- semilla sintética `20261020`;
- siete repeticiones por algoritmo y carga, alternando el orden.

Cada repetición se ejecuta en un proceso independiente. El protocolo separa
importación del módulo, carga o generación de datos, preparación, primera
selección, selección caliente, tiempo del proceso y RSS máximo. Antes de la
campaña se mide una compilación sin caché y una primera llamada con caché para
`float32` y `float64`.

Solo `datawave` es un dataset real incluido en el repositorio. Las cargas
Franklin, Castro, NC State y TU Delft reproducen tamaños publicados mediante
datos sintéticos; no son reproducciones de esos datasets.

## Resultados principales

La tabla muestra medianas de selección. La columna Numba corresponde a la
segunda llamada del mismo proceso, después de cargar o compilar el núcleo.

| Carga | Filas | Selecciones | 2.2 (s) | Numba (s) | Aceleración | Ahorro | Cambio RSS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `datawave` | 140.256 | 20 | 0,056 | 0,013 | 4,45× | 0,044 s | +2,5 % |
| `datawave` | 140.256 | 100 | 0,254 | 0,076 | 3,34× | 0,178 s | +2,3 % |
| Franklin et al. | 87.664 | 600 | 0,511 | 0,165 | 3,09× | 0,346 s | 0,0 % |
| Castro et al. | 121.968 | 100 | 0,123 | 0,036 | 3,40× | 0,087 s | +0,6 % |
| Castro et al. | 121.968 | 1.000 | 1,375 | 0,414 | 3,32× | 0,960 s | +1,5 % |
| Castro et al. | 121.968 | 5.000 | 6,423 | 1,968 | 3,26× | 4,455 s | +0,8 % |
| NC State | 1.000.000 | 1.250 | 12,839 | 3,942 | 3,26× | 8,897 s | +13,4 % |
| TU Delft | 10.000 | 1.000 | 0,102 | 0,030 | 3,36× | 0,072 s | 0,0 % |
| Control direccional | 87.664 | 600 | 0,962 | 0,389 | 2,47× | 0,573 s | 0,0 % |

La matriz de escalado mantuvo mejoras en todas sus dimensiones. La aceleración
mínima fue 1,37× con una columna y la máxima 4,77× con `float32`.
`float64` mejoró 3,86×. Con todas las columnas
direccionales la mejora se redujo a 2,11×, pero siguió siendo positiva.

## Compilación, caché y punto de equilibrio

La primera compilación costó aproximadamente 0,57 segundos en `float64` y 0,44
segundos en `float32`. Con código ya almacenado, cargar la especialización y
ejecutar la primera selección añadió aproximadamente 0,10 segundos; una llamada
caliente posterior eliminó ese coste. Importar el candidato añadió entre 0,08 y
0,12 segundos frente a la referencia 2.2 en las cargas principales.

Considerando importación, preparación y selección, el punto de equilibrio
estimado en un proceso que reutiliza los mismos datos es:

| Carga | Caché disponible | Primera compilación |
| :--- | ---: | ---: |
| `datawave`, 20 selecciones | 4 llamadas | 15 llamadas |
| `datawave`, 100 selecciones | 2 llamadas | 4 llamadas |
| Franklin, 600 selecciones | 1 llamada | 3 llamadas |
| Castro y NC State | 1 llamada | 1 llamada |
| Control direccional | 1 llamada | 2 llamadas |

Por tanto, Numba no conviene para llamadas pequeñas y aisladas, pero amortiza
su coste dentro de la primera llamada en las cargas grandes.

## Memoria e instalación

El RSS mediano del proceso Numba aumentó entre 0 % y 13,4 % en las cargas
principales. En la matriz de escalado, los mayores aumentos fueron 12,7 % con un
millón de filas y 23,4 % con 30 columnas. El núcleo evita los dos buffers
bidimensionales de diferencias y valores normalizados de 2.2, pero el runtime
de Numba/LLVM eleva la base del proceso. La complejidad espacial del algoritmo
no aumenta.

La instalación temporal ocupó 16,24 MiB para Numba y 167,81 MiB para llvmlite,
184,05 MiB en conjunto. NumPy ya era una dependencia de `mdapy` y no se suma a
esa cifra. Este coste confirma que Numba solo tendría sentido como extra
opcional y con importación diferida.

## Decisión para la siguiente subfase

La evaluación secuencial queda aceptada desde el punto de vista algorítmico y
de rendimiento. El paralelismo se estudió después en una rama y una campaña
independientes para no alterar esta referencia.

Antes de incorporar Numba se requiere una subfase independiente que:

1. repita corrección, primera llamada y ejecución caliente al menos en Linux
   x86-64, macOS ARM64 y Windows x86-64;
2. determine el alcance del fallo Linux ARM64/WSL2 y si una versión posterior
   de Numba lo corrige;
3. acuerde la matriz compatible de Python, NumPy y Numba;
4. solo después diseñe el extra de instalación y la selección explícita del
   motor, manteniendo NumPy como predeterminado.

## Reproducción

La validación compacta se ejecuta sin cambiar las dependencias del proyecto:

```bash
NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.numba.evaluate --check
```

Las campañas de confirmación y escalado se reproducen con:

```bash
NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.numba.evaluate \
  --suite grounded --output benchmarks/experiments/numba/results/grounded

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.numba.evaluate \
  --suite scaling --output benchmarks/experiments/numba/results/scaling
```

Sus archivos `metadata.json` registran versiones, plataforma, semilla, cargas,
repeticiones y hashes de ambas referencias; `compilation.json` conserva los
costes fríos.
