# Análisis de rendimiento

Esta carpeta permite estudiar cómo evoluciona el algoritmo de máxima
disimilitud. No sigue automáticamente el código en desarrollo: cada referencia
se añade cuando existe una evolución publicada que merece ser comparada.

## Organización

- `algorithms/vX_Y.py` conserva el último parche disponible de una evolución.
- `cases.json` fija los índices esperados para cada referencia.
- `run.py` valida los casos y ejecuta las mediciones.
- `RESULTS.md` reúne las comparaciones interpretadas.
- `results/` conserva los datos brutos de las campañas documentadas.

Git conserva las revisiones anteriores. Si aparece un parche posterior de una
serie, se actualiza su única referencia en lugar de añadir otra copia.

## Medición

Cada repetición se ejecuta en un proceso independiente. El runner registra las
versiones del entorno, hashes de los datos y referencias, tamaño y dtype de la
entrada, semilla, índices seleccionados, tiempos de preparación y selección,
tiempo total del proceso y memoria RSS máxima.

Antes de medir, todas las referencias deben superar sus casos. Las diferencias
de índices entre evoluciones se declaran explícitamente; no se ocultan mediante
tolerancias ni actualizando una expectativa sin justificarla.

Validar los casos:

```bash
uv run benchmarks/run.py --check
```

Ejecutar la campaña configurada:

```bash
uv run benchmarks/run.py
```

Ejecutar una exploración con cargas distintas:

```bash
uv run benchmarks/run.py --counts 5 10 --runs 2 --timeout 60
```

El runner crea por defecto un directorio con fecha y hora. Para dar un nombre
estable a una campaña se puede indicar su destino:

```bash
uv run benchmarks/run.py --output benchmarks/results/v2_1-v2_2-AAAA-MM-DD
```

Solo se versionan los directorios que respaldan una entrada de `RESULTS.md`;
las ejecuciones exploratorias se eliminan después de revisarlas. Cada entrada
incluye el comando utilizado y enlaza sus datos brutos.
