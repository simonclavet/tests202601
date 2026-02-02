// Precompiled header for Flomo
// Heavy headers that rarely change go here

#pragma once

// Standard C headers
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <float.h>
#include <errno.h>

// Windows headers must come before raylib to avoid conflicts
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOGDI             // Excludes GDI (avoids Rectangle conflict)
#define NOUSER            // Excludes USER (avoids CloseWindow, ShowCursor, DrawText conflicts)
#include <windows.h>
#undef near               // Legacy 16-bit segment pointer macros - begone
#undef far
#endif

// Raylib core headers
#include "raylib.h"
#include "rcamera.h"
#include "raymath.h"
#include "rlgl.h"

// Suppress torch warnings
#pragma warning(push)
#pragma warning(disable: 4267)  // size_t to int conversion
#pragma warning(disable: 4251)  // DLL interface issues
#pragma warning(disable: 4275)  // non-DLL-interface
#pragma warning(disable: 4996)  // deprecation warnings
#pragma warning(disable: 4702)  // unreachable code (from irange.h)

#include <torch/torch.h>

#pragma warning(pop)

// Standard library headers commonly used
#include <vector>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <cstring>

#include <cstdio>
#include <cstdlib>
#include <fstream>
