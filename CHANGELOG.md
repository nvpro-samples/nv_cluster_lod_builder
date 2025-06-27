
# Version 2

## Features

- Shared library support in cmake, [`NVCLUSTERLOD_BUILDER_SHARED`](CMakeLists.txt)

## Code Quality

- Real C API, removing namespace, adding prefixes, symbol export
- Triangles now vec3u rather than indices
- Spheres use vec3f center
- Removed `vertexOffset`
- Fallback for missing libc++ parallel execution
