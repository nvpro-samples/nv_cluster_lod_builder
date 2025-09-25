# Changelog

## [4]

### Added

- Optional decimation callback

## [3]

### Added

- Support `nvclusterlod_ContextCreateInfo::parallelize`.

### Changed

- Modified error enums.
- Use `std::span` internally.

## [2]

### Added

- Shared library support in CMake (`NVCLUSTERLOD_BUILDER_SHARED`).
- Fallback for missing libc++ parallel execution.

### Changed

- Real C API, removing namespace, adding prefixes, and symbol export.
- Triangles now `vec3u` rather than indices.
- Spheres use `vec3f` center.

### Removed

- `vertexOffset` input parameter.
