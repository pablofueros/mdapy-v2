# Experimentos de rendimiento

Esta carpeta conserva candidatos ejecutables que reutilizan los casos, datos y
cargas de `benchmarks`, pero que todavía no forman parte de una versión oficial
de `mdapy`.

Cada experimento debe mantener juntos su implementación, protocolo, resultados
y limitaciones. Si un candidato se incorpora a la biblioteca, su implementación
pasa a `src/mdapy` y la evolución aceptada se añade a las comparaciones
consolidadas de `benchmarks/RESULTS.md`.

Los experimentos no son API pública ni dependencias del paquete.

## Resultados compactos

Las campañas exploratorias generan muchas más combinaciones que los benchmarks
oficiales. Por ello, cada directorio de campaña conserva un formato compacto:

- `campaign.json` reúne metadatos, sondas de compilación y resumen;
- `measurements.jsonl` agrupa todas las repeticiones de cada combinación;
- `validations.jsonl` guarda cada secuencia de índices una sola vez, identificada
  por su hash SHA-256.

Los evaluadores comprueban que el formato compacto reconstruye exactamente los
registros originales antes de retirar los archivos redundantes. Esta decisión
no cambia el formato de `benchmarks/results`, cuyas campañas oficiales son más
pequeñas y mantienen su estructura histórica mientras siga siendo manejable.

## Evaluaciones disponibles

- [`numba/SEQUENTIAL.md`](numba/SEQUENTIAL.md) y
  [`numba/PARALLEL.md`](numba/PARALLEL.md) estudian compilación JIT secuencial
  y paralela.
