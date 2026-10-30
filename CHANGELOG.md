# Changelog

## [2.3.1](https://github.com/pablofueros/mdapy-v2/compare/v2.3.0...v2.3.1) (2026-10-30)


### Bug Fixes

* remove downloads badge from README ([e19f4eb](https://github.com/pablofueros/mdapy-v2/commit/e19f4eb978bb860257756240931abdccbb6b33b7))

## [2.3.0](https://github.com/pablofueros/mdapy-v2/compare/v2.2.1...v2.3.0) (2026-10-30)


### Features

* add optional Numba engine ([74d616f](https://github.com/pablofueros/mdapy-v2/commit/74d616ff7955286d62212932d4ae5d35c8f4ea2e))


### Bug Fixes

* preserve progress API compatibility ([bff1e89](https://github.com/pablofueros/mdapy-v2/commit/bff1e89a3e43e25f7813b75fe6f83a16b1e6093d))
* standardize dtype calculations ([99f0d6a](https://github.com/pablofueros/mdapy-v2/commit/99f0d6aaccbfa5c456bc361f11021b01a075e300))


### Performance Improvements

* confirm public Numba engine ([142a837](https://github.com/pablofueros/mdapy-v2/commit/142a8377de1e8953afaad91d85393264faabd326))


### Code Refactoring

* remove obsolete progress output ([82f0f25](https://github.com/pablofueros/mdapy-v2/commit/82f0f250cd5d46e19b089e719bfdf4a485a119c4))
* simplify numba adapter ([202ff30](https://github.com/pablofueros/mdapy-v2/commit/202ff30ad5695ad21a9844c186e9248948132cf6))

## [2.2.1](https://github.com/pablofueros/mdapy-v2/compare/v2.2.0...v2.2.1) (2026-10-27)


### Bug Fixes

* use absolute performance image URL ([6cf7191](https://github.com/pablofueros/mdapy-v2/commit/6cf7191480a7ab49a4db1b41ea9ed463cc92f00a))

## [2.2.0](https://github.com/pablofueros/mdapy-v2/compare/v2.1.0...v2.2.0) (2026-10-27)


### Bug Fixes

* satisfy benchmark type checking ([bf2fe47](https://github.com/pablofueros/mdapy-v2/commit/bf2fe47d986fe4379ce14e18b3600289ba84bc71))


### Performance Improvements

* prototype Python and NumPy distance kernel ([cb9bd0e](https://github.com/pablofueros/mdapy-v2/commit/cb9bd0e1a9c29cb1b80afa537ecd968f416a4eeb))


### Code Refactoring

* clarify NumPy distance buffers ([5473e85](https://github.com/pablofueros/mdapy-v2/commit/5473e8534d11f9c648a544e997d041e98f1e750a))


### Miscellaneous Chores

* prepare release 2.2.0 ([157e681](https://github.com/pablofueros/mdapy-v2/commit/157e6813e49c6e505df196143c14f6c5862392ae))

## [2.1.0](https://github.com/pablofueros/mdapy-v2/compare/v2.0.0...v2.1.0) (2026-10-17)


### Features

* finalize column-based API ([3100329](https://github.com/pablofueros/mdapy-v2/commit/3100329d2bcb031498386372798c5ab892e3195e))
* migrate visual examples to Plotly ([3056cd3](https://github.com/pablofueros/mdapy-v2/commit/3056cd36508f9a5e66b185d75e5d1529e92bbe0c))
* preserve input precision in selection ([243cda6](https://github.com/pablofueros/mdapy-v2/commit/243cda61d097fdb0cb2e4a6467bda94fb7ae7f0c))
* update Python support and development tooling ([6be1345](https://github.com/pablofueros/mdapy-v2/commit/6be13454f2164d3b8e8eacc9e047e4fb263f6933))


### Code Refactoring

* rename DataColumn array property ([0db11ba](https://github.com/pablofueros/mdapy-v2/commit/0db11ba3831b61ab8b05234847f2fecec229acbe))
* rename public API module ([6d7046b](https://github.com/pablofueros/mdapy-v2/commit/6d7046bce91953dfdcfab5c0c8b74b879384568b))

## [2.0.0](https://github.com/pablofueros/mdapy-v2/compare/v0.1.1...v2.0.0) (2026-10-13)

First release of the maintained `mdapy-v2` distribution. The distribution
name changes from `ihc-mdapy`, while the Python import remains `mdapy`.

### Features

- Introduce the `DataColumn` API for declaring scalar and directional variables.
- Keep the tuple-based API available explicitly through `mdapy.legacy`.
- Add validation for column shapes, finite values, cluster counts, seed indices,
  and undefined deviations.

### Included fixes

- Include the NaN validation contributed by Álvaro Rodríguez in the historical
  but unpublished `v0.1.1` tag.
