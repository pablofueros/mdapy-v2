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

Cada directorio de campaña conserva el mismo formato compacto que los
benchmarks oficiales:

- `campaign.json` reúne metadatos, sondas de compilación y resumen;
- `measurements.jsonl` agrupa todas las repeticiones de cada combinación;
- `validations.jsonl` guarda cada secuencia de índices una sola vez, identificada
  por su hash SHA-256.

Los evaluadores comprueban que el formato compacto reconstruye exactamente los
registros originales antes de retirar los archivos redundantes. Esta decisión
evita duplicar los índices seleccionados sin perder muestras, metadatos ni
resultados históricos.

## Evaluaciones disponibles

- [`numba/SEQUENTIAL.md`](numba/SEQUENTIAL.md) y
  [`numba/PARALLEL.md`](numba/PARALLEL.md) estudian compilación JIT secuencial
  y paralela;
- [`rust/SEQUENTIAL.md`](rust/SEQUENTIAL.md) y
  [`rust/PARALLEL.md`](rust/PARALLEL.md) estudian núcleos Rust ejecutados
  mediante `rust-script`, sin integrarlos todavía con Python.
