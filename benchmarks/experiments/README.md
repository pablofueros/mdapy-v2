# Experimentos de rendimiento

Esta carpeta conserva candidatos ejecutables que reutilizan los casos, datos y
cargas de `benchmarks`, pero que todavía no forman parte de una versión oficial
de `mdapy`.

Cada experimento debe mantener juntos su implementación, protocolo, resultados
y limitaciones. Si un candidato se incorpora a la biblioteca, su implementación
pasa a `src/mdapy` y la evolución aceptada se añade a las comparaciones
consolidadas de `benchmarks/RESULTS.md`.

Los experimentos no son API pública ni dependencias del paquete.
