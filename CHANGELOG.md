# Changelog

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
