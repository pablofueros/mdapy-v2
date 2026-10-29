# Evaluación del paralelismo de Numba para `mdapy` 2.3

## Alcance y conclusión

Esta campaña compara el núcleo NumPy 2.2, el candidato Numba secuencial y una
variante que paraleliza por filas la actualización de distancias. No cambia la
API pública ni las dependencias del paquete.

El paralelismo es una mejora importante para las cargas pesadas a las que se
dirigiría un motor Numba opcional, aunque no acelera cualquier carga posible:

- reprodujo exactamente los mismos índices y su orden en todos los casos de
  referencia, los cinco casos de ajuste, las nueve cargas de confirmación y las
  19 cargas de escalado;
- con cuatro hilos mejoró las nueve cargas de confirmación entre 1,79× y 3,63×
  respecto a Numba secuencial, y entre 6,89× y 11,41× respecto a NumPy 2.2;
- mejoró 12 de las 19 cargas de escalado frente al candidato secuencial, con
  una mediana de 1,92×, pero empeoró siete cargas pequeñas;
- una carga de una sola columna fue 0,59× respecto a NumPy y 0,43× respecto a
  Numba secuencial: el coste de coordinar hilos supera el trabajo disponible.

Estos resultados deben interpretarse dentro del contrato de uso previsto.
NumPy continuaría siendo el motor predeterminado para cargas normales, sin
dependencias ni configuración adicionales. Numba sería una elección explícita
para una carga grande o para varias ejecuciones que amorticen la compilación.
En ese ámbito, el kernel paralelo es el candidato principal.

Una futura integración no debería ofrecer dos motores Numba. El kernel
secuencial puede conservarse como referencia o como respaldo interno para
casos evidentemente pequeños, pero una selección adaptativa perfecta no es un
requisito para justificar el motor acelerado.

## Diseño del candidato

La dependencia entre selecciones permanece serial: cada nueva observación
depende de las distancias mínimas calculadas en las iteraciones anteriores. En
cada iteración, `prange` distribuye por filas el cálculo de distancias y su
actualización. La búsqueda del máximo se mantiene serial para conservar de forma
explícita el desempate de 2.2 a favor del primer índice.

No se usa `fastmath` ni se paralizan reducciones numéricas. Esta estructura
evita que el orden de los hilos altere las sumas por observación o la selección.
El ejecutor compara los índices tras cada variante y cancela la campaña ante la
primera diferencia.

## Ajuste del número de hilos

Se midieron 1, 2, 4, 8 y 12 hilos sobre cinco cargas representativas, con siete
repeticiones por configuración. La tabla resume la aceleración de la selección
caliente frente al candidato Numba secuencial:

| Hilos | Mediana | Mínimo | Máximo | Cargas más rápidas |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,07× | 0,98× | 1,16× | 3/5 |
| 2 | 1,58× | 1,43× | 2,06× | 5/5 |
| 4 | 2,31× | 1,31× | 3,74× | 5/5 |
| 8 | 2,31× | 1,04× | 5,63× | 5/5 |
| 12 | 2,04× | 0,95× | 4,77× | 4/5 |

Cuatro hilos obtuvieron la mejor mediana y un mínimo claramente superior al de
ocho hilos. Se eligieron para las comparaciones finales de este equipo, no como
valor predeterminado para otras máquinas. El backend utilizado fue OpenMP.

## Confirmación con cuatro hilos

La tabla muestra medianas de selección caliente. La aceleración adicional toma
como referencia el candidato Numba secuencial de la campaña actual.

| Carga | Selecciones | NumPy (s) | Numba sec. (s) | Numba par. (s) | Adicional | Frente a NumPy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `datawave` | 20 | 0,062 | 0,016 | 0,009 | 1,79× | 6,89× |
| `datawave` | 100 | 0,306 | 0,085 | 0,027 | 3,15× | 11,41× |
| Franklin et al. | 600 | 0,618 | 0,195 | 0,068 | 2,85× | 9,04× |
| Castro et al. | 100 | 0,144 | 0,046 | 0,017 | 2,75× | 8,54× |
| Castro et al. | 1.000 | 1,468 | 0,481 | 0,161 | 2,98× | 9,09× |
| Castro et al. | 5.000 | 7,813 | 2,502 | 0,919 | 2,72× | 8,50× |
| NC State | 1.250 | 15,876 | 4,755 | 2,011 | 2,36× | 7,89× |
| TU Delft | 1.000 | 0,149 | 0,039 | 0,016 | 2,38× | 9,06× |
| Control direccional | 600 | 1,261 | 0,480 | 0,132 | 3,63× | 9,53× |

El mayor ahorro adicional fue 2,74 segundos por llamada en NC State. En Castro
con 5.000 selecciones se ahorraron 1,58 segundos adicionales. La memoria máxima
fue similar a la del candidato secuencial en estas cargas.

## Límites observados en el escalado

La comparación de 19 cargas produjo estos rangos con cuatro hilos:

| Referencia | Mínimo | Mediana | Máximo | Cargas más rápidas |
| :--- | ---: | ---: | ---: | ---: |
| Numba secuencial | 0,31× | 1,92× | 3,94× | 12/19 |
| NumPy 2.2 | 0,59× | 5,40× | 10,86× | 18/19 |

Las regresiones frente al candidato secuencial aparecieron con cinco
selecciones, 10.000 o 100.000 filas, una o tres columnas y otras combinaciones
con poco trabajo por iteración. No indican un error numérico: son el coste fijo
de planificar y sincronizar el bucle paralelo. A partir de cargas mayores, más
columnas o más selecciones, el paralelismo pasa a ser favorable.

Estas regresiones delimitan el ámbito del motor opcional, pero no invalidan su
uso para trabajo pesado: las cargas pequeñas continuarían usando NumPy de forma
predeterminada. Si se añade un respaldo secuencial interno, no basta con usar
solo el número de filas como umbral. El trabajo depende también de columnas,
selecciones y columnas direccionales, y la frontera cambia con CPU y backend de
hilos.

## Coste frío y compatibilidad

El candidato paralelo necesita otra especialización compilada. En este entorno,
la compilación sin caché costó aproximadamente 1,03 segundos para `float64` y
0,85 segundos para `float32`, frente a 0,80 y 0,51 segundos del candidato
secuencial en la misma campaña. Con caché, la primera selección de ambos quedó
alrededor de 0,11 segundos en las sondas pequeñas.

Las mediciones se ejecutaron con Numba 0.68.1 y llvmlite 0.50.0 sobre Linux
ARM64 bajo WSL2, 12 CPU y `NUMBA_CPU_NAME=generic`. Sin esa configuración, LLVM
aborta durante la generación de código nativo en este equipo. Esta limitación es
la misma detectada por la evaluación secuencial y sigue impidiendo recomendar
la integración antes de probar las plataformas objetivo.

## Decisión experimental

Numba queda validado como vía educativa y como optimización potencial para
cargas pesadas o repetidas. NumPy seguiría siendo la opción sencilla y
predeterminada. Dentro del motor Numba, el candidato paralelo ofrece el mejor
rendimiento para el uso previsto; el secuencial conserva valor como referencia
y posible respaldo interno. El control del número de hilos sí debe formar parte
del diseño, pero el despacho automático entre ambos kernels es una mejora
posterior, no una condición para considerar viable el motor.

La siguiente subfase, si se decide integrar, debe mantenerse separada de este
experimento y resolver primero:

1. compatibilidad y corrección en Linux x86-64, macOS ARM64 y Windows x86-64;
2. el alcance documentado del motor para cargas grandes o repetidas y, de forma
   opcional, un respaldo conservador para casos claramente pequeños;
3. convivencia con aplicaciones que ya administran sus propios hilos;
4. extra de instalación, importación diferida y fallback claro a NumPy;
5. contrato del futuro `engine`, sin exponer los dos kernels Numba como motores
   públicos distintos.

## Reproducción

El ajuste y las dos comparaciones se reproducen sin modificar las dependencias
del proyecto:

```bash
NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.numba.evaluate_parallel \
  --mode tuning --runs 7 \
  --thread-counts 1 2 4 8 12 \
  --output benchmarks/experiments/numba/results/parallel/tuning

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.numba.evaluate_parallel \
  --mode comparison --suite grounded --threads 4 --runs 7 \
  --output benchmarks/experiments/numba/results/parallel/grounded-t4

NUMBA_CPU_NAME=generic uv run --with 'numba>=0.68,<1' \
  python -m benchmarks.experiments.numba.evaluate_parallel \
  --mode comparison --suite scaling --threads 4 --runs 7 \
  --output benchmarks/experiments/numba/results/parallel/scaling-t4
```

Cada directorio usa el [formato compacto de experimentos](../README.md#resultados-compactos)
para conservar metadatos, hashes, sondas de compilación, todas las muestras,
índices exactos y el resumen sin repetir información por ejecución.
