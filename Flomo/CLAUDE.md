# Flomo Project Guidelines

## Project Overview
Flomo is a BVH/FBX animation file viewer with CUDA and LibTorch integration. Originally ported from orangeduck's BVHView (C) to C++, using raylib for rendering and raygui for UI.

## Features
- Load and view BVH animation files
- Load and view FBX animation files (via ufbx library)
- Convert FBX to BVH via command line: `flomo.exe -fbx2bvh input.fbx`
- CUDA acceleration
- LibTorch integration (for future ML features)

## Build System
- **CMake** (minimum 3.21)
- **C++20** standard, **CUDA 17**
- **Visual Studio 2022** generator on Windows
- Dependencies in `thirdparty/` (no downloads during build)

### Build Commands
```bash
# Configure (from project root)
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build Debug
cmake --build . --config Debug

# Build Release
cmake --build . --config RelWithDebInfo

# Output: build/src/Debug/flomo.exe or build/src/RelWithDebInfo/flomo.exe
```

### Claude Code Build Command
Use forward slashes for paths in bash:
```bash
cd "F:/experiments/tests202601/Flomo/build" && cmake --build . --config RelWithDebInfo
```

### Command Line Usage
```bash
# View animation files (GUI mode)
flomo.exe file.bvh
flomo.exe file.fbx

# Convert FBX to BVH (headless)
flomo.exe -fbx2bvh input.fbx
# Creates: input.fbx.bvh
```

## Code Style Preferences

Here are some guidelines.
Some of these conventions are not really respected everywhere in this code, but try to follow them as much as 
possible for any new - Prefer `int` as the main integer type (don't warn about size_t to int conversions)
- Use `float` for all floating-point values (add `f` suffix to literals: `1.0f`)
- Be const-correct for pointer/reference parameters and local variables
- C++20 designated initializers are OK
- Keep code in single files when practical
code you write.
When possible initialize variables directly when defining them in the struct definition instead of in the init function or constructor.
Function names are CamelCase, variable and member names are camelCase. constexpr for CONSTANTS
don't use auto unless absolutely necessary.
don't use lambdas
be const correct for functions and function parameters (except simple parameter values). 
be const correct for local variable: if a variable is not modified after initialization, make it const.
don't create unused variables
When an argument is modified as the result, prefix the argument with /*out*/ if it is not obvious.
never modify non-const value arguments of functions.
don't use pairs and tuples, make small structs instead
don't use smart pointers
don't use complicated oop concepts like inheritance and polymorphism, unless absolutely necessary
don't use exceptions, use assertions defensively. Don't use the keyword noexcept, catch, try, unless absolutely necessary
use linebreaks before opening braces for functions and control blocks, always use braces for if/for/while, except when it is verysimple single-line statements
when writing comments, be casual, no need for things like ---- and other heading decorations like numbers or letters for steps
no need for private and public. Use structs only
Try to keep things simple. If you spot opportunities for removing abstractions, deadcode, unecessary complications, tell the human about it.
ask questions to the human if unsure about anything. Don't assume things, ask instead.
be casual when conversing with the human. We want to have fun coding together. 

When you are copilot and you don't modify code but tell me changes to do, please don't copy paste the whole function or file, 
unless it is small. Instead tell me clearly what to change, with a bit of context before and after the new thing. 
Don't put + and - diff markers. 

## Compiler Warnings Policy
- Warnings as errors (`/WX`)
- Warning level 4 on MSVC (`/W4`)
- **Suppressed warnings:** 4100, 4267, 4018, 4505

## Dependencies (in thirdparty/)
- **raylib** - Graphics/window library
- **raygui** - Immediate mode GUI (header-only)
- **ufbx** - FBX file loading library
- **libtorch** - PyTorch C++ API (separate debug/release folders)

## File Structure
```
Flomo/
├── CMakeLists.txt           # Root CMake (project config, adds subdirs)
├── src/
│   ├── CMakeLists.txt       # Flomo target definition
│   ├── flomo.cpp            # Main application
│   ├── cuda_kernels.cu      # CUDA compute kernels
│   ├── bvh_parser.h         # BVH file loading/saving
│   ├── fbx_loader.h         # FBX file loading (uses ufbx)
│   ├── transform_data.h     # Animation transform handling
│   └── ... (other headers)
├── thirdparty/
│   ├── raylib/
│   ├── raygui/
│   ├── ufbx/
│   └── libtorch/ (debug/ and release/)
├── data/                    # Sample animation files
├── shaders/                 # GLSL shaders
├── build/                   # Build output (gitignored)
└── FBX2BVH/                 # Legacy (now integrated into flomo)
```

## Key Implementation Notes

### FBX Loading (fbx_loader.h)
- Uses ufbx library with `target_unit_meters=1.0` and `target_axes=right_handed_y_up`
- Detects and skips static "Reference" nodes (no translation animation)
- Unit scaling: `node_to_world` is meters, `node_to_parent` is original units (cm)

### BVH Format
- Euler rotation order: ZXY (R = Rz * Rx * Ry)
- Root has 6 channels (position + rotation), other joints have 3 (rotation only)
- End sites have no channels

### CUDA Settings
- Architecture: SM 86 (RTX 30 series)
- Flags: `--extended-lambda`, `--expt-relaxed-constexpr`, `-lineinfo`


plan for jan 29:
posvel rep for anims
motion matching