<div align="center">

<img src="https://raw.githubusercontent.com/pablofueros/mdapy-v2/main/assets/banner.png" alt="mdapy logo" width="700"/>

---

### ⚡ **FAST Maximum Dissimilarity Algorithm in Python** ⚡

[![Code Quality](https://github.com/pablofueros/mdapy-v2/actions/workflows/code-quality.yml/badge.svg)](https://github.com/pablofueros/mdapy-v2/actions/workflows/code-quality.yml)
[![Coverage](https://codecov.io/gh/pablofueros/mdapy-v2/graph/badge.svg?branch=main)](https://codecov.io/gh/pablofueros/mdapy-v2)
[![PyPI Latest Release](https://img.shields.io/pypi/v/mdapy-v2.svg)](https://pypi.org/project/mdapy-v2/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mdapy-v2.svg)](https://pypi.org/project/mdapy-v2/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/pablofueros/mdapy-v2.svg)](https://github.com/pablofueros/mdapy-v2/blob/main/LICENSE)

---

[![mdapy-v2 2.2 completes the benchmark in 0.067 seconds compared with 32.63
seconds for ihc-mdapy](https://raw.githubusercontent.com/pablofueros/mdapy-v2/main/assets/performance/v2_0-v2_2.svg)](#performance)

</div>

`mdapy-v2` is a maintained fork of
[`IHCantabria/mdapy`](https://github.com/IHCantabria/mdapy), developed by the
original author. It keeps the legacy `mdapy` Python import while evolving the
public API, correctness checks, examples, and performance of the algorithm.

> [!IMPORTANT]
> The distributions `mdapy-v2`, `ihc-mdapy`, and the unrelated `mdapy` project
> all install a package named `mdapy`. Treat them as alternatives and do not
> install them together in the same Python environment.

## Installation

`mdapy-v2` requires Python 3.12 or newer. [`uv`](https://docs.astral.sh/uv/) is also recommended.

```bash
uv pip install mdapy-v2
```

Install the optional Numba engine if you want to accelerate large or repeated
selections:

```bash
uv pip install "mdapy-v2[numba]"
```

The distribution name changed, but the import remains short and familiar:

```python
from mdapy import DataColumn, max_diss_alg
```

## Quick start

Each `DataColumn` represents one variable. Mark angular variables as
directional so that, for example, the distance between 350° and 10° is 20°
rather than 340°.

```python
from mdapy import DataColumn, max_diss_alg

tp = [14.93, 10.42, 7.46, 7.63, 12.20, 12.05, 14.49, 8.62]
hs = [1.82, 1.45, 0.26, 0.90, 1.30, 5.71, 1.40, 2.18]
direction = [268, 272, 289, 293, 276, 273, 276, 292]

points, indices = max_diss_alg(
    DataColumn(tp),
    DataColumn(hs),
    DataColumn(direction, directional=True),
    n_clusters=3,
)
```

`points` contains the selected observations in selection order and `indices`
contains their positions in the original input. Use `seed_index` to choose a
different initial observation:

```python
points, indices = max_diss_alg(
    DataColumn(tp),
    DataColumn(hs),
    n_clusters=3,
    seed_index=4,
)
```

### Optional Numba engine

After installing the `numba` extra, select the compiled engine explicitly:

```python
points, indices = max_diss_alg(
    DataColumn(tp),
    DataColumn(hs),
    n_clusters=3,
    engine="numba",
)
```

NumPy remains the default. See [Numba acceleration](#numba-acceleration) for
performance guidance.

## How it works

The implementation follows the formulation described in
[Appendix 1 of Guanche et al. (2013)](#reference).

The Maximum Dissimilarity Algorithm builds a diverse subset iteratively:

1. Start with the observation at `seed_index` (position `0` by default).
2. Measure every unselected observation against its nearest selected one.
3. Select the observation whose nearest distance is greatest.
4. Repeat until `n_clusters` observations have been selected.

Distances are Euclidean after normalizing each variable by its standard
deviation. Directional variables use the shortest angular distance and a
circular standard deviation. Exact ties preserve input order.

## Performance

Benchmarks require exact selected indices before recording timings. The chart
at the top labels the preserved 2.0 reference as `ihc-mdapy` because it uses
the same selection algorithm as the latest published release, adapted to the
current API. It compares that reference with the default NumPy engine introduced
in version 2.2 on the same `datawave` workload and environment.

Most of the improvement came from maintaining incremental minimum distances
instead of repeatedly scanning the selected subset and growing a pair cache.
Version 2.2 then reduced the cost of each selection by reusing buffers and
comparing squared distances. With 140,256 observations and 20 selections, the
median complete call decreased from 32.63 seconds to 0.067 seconds.

Version 2.3 keeps this NumPy engine as the default and adds Numba as an explicit
option for workloads where saving seconds justifies its compilation cost. These
measurements describe one workload and environment, not a universal speedup.
See the [benchmark results](benchmarks/RESULTS.md) for the complete protocol,
reproduction commands, and raw data.

## Numba acceleration

NumPy remains the best starting point for small or one-off selections. Choose
Numba explicitly when a selection already takes seconds or when a large
workload is repeated. `mdapy` does not switch automatically because the payoff
depends on the number of rows, variables and selections, as well as the CPU.

The first call for each input dtype compiles or loads a cached specialization;
later calls reuse that compiled code. `mdapy` also leaves the thread count to
Numba instead of changing an application's thread configuration.

<p align="center">
  <a href="benchmarks/RESULTS.md">
    <img src="https://raw.githubusercontent.com/pablofueros/mdapy-v2/main/assets/performance/numpy-numba.svg" alt="NumPy completes the large selection in 6.45 seconds, compared with 1.72 seconds for the first Numba call with its cache available and 0.85 seconds for a repeated Numba call" width="760">
  </a>
</p>

The public-API campaign used `float64` inputs, and the first Numba call had its
disk cache available. Both engines selected exactly the same indices. Numba
used 5.2% more process memory, so this is a speed optimization rather than a
memory optimization. Timings are machine-specific; the
[benchmark results](benchmarks/RESULTS.md) contain the complete protocol and
raw measurements.

The [Numba example](examples/5_mda_numba.py) recreates this deterministic
workload, times both engines and checks their results:

```bash
uv run --extra numba python examples/5_mda_numba.py
```

## Examples

Clone the repository and run any example in the project environment:

```bash
uv run python examples/1_mda_minimal.py
```

Start with the [minimal selection](examples/1_mda_minimal.py), a one-dimensional
case small enough to follow in a debugger. The next example animates how the
algorithm expands a diverse subset through a two-dimensional sample.

<p align="center">
  <a href="examples/2_mda_selection.py">
    <img src="https://raw.githubusercontent.com/pablofueros/mdapy-v2/main/assets/examples/selection.png" alt="Maximum dissimilarity selection order in a two-dimensional sample" width="850">
  </a>
</p>

For a focused explanation of circular distances, see the
[directional variables example](examples/3_mda_directional.py). The fourth
example applies the algorithm to select representative sea states from 10,000
directional observations.

<p align="center">
  <a href="examples/4_mda_datawave.py">
    <img src="https://raw.githubusercontent.com/pablofueros/mdapy-v2/main/assets/examples/datawave.png" alt="Representative sea states selected from directional wave data" width="750">
  </a>
</p>

## Legacy tuple API

Projects migrating from the original tuple-based API can use an explicit
compatibility import:

```python
from mdapy.legacy import max_diss_alg

points, indices = max_diss_alg(
    (tp, hs, direction),
    3,
    seed_index=0,
    dir_indices=[2],
)
```

Passing a tuple to `mdapy.max_diss_alg` raises an error that points to this
compatibility module. The published `ihc-mdapy==0.1.0` package uses the legacy
API; its unpublished `v0.1.1` fix remains available in this repository's Git
history.

## Development

Install the locked development environment and run the project checks:

```bash
make install
make format
make lint
make test
```

## Credits and license

Originally developed by [Pablo Garcia](https://github.com/pablofueros) at
[IHCantabria](https://github.com/IHCantabria) and now maintained as the
personal `mdapy-v2` fork. The historical `v0.1.1` fix was contributed by
[Álvaro Rodríguez](https://github.com/arodriguezluis), but was never included
in the published `ihc-mdapy` package.

The original IHCantabria copyright notice and MIT terms are preserved in
[LICENSE](LICENSE).

## Reference

Guanche, Y., Guanche, R., Camus, P., Mendez, F. J., & Medina, R. (2013).
“A multivariate approach to estimate design loads for offshore wind turbines.”
*Wind Energy*, 16(7), 1091–1106.
[https://doi.org/10.1002/we.1542](https://doi.org/10.1002/we.1542)
