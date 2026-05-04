# Hash-Encoded Neural SDF Training + Rendering

A small application written in CUDA using tiny-cuda-nn and OpenGL for visualizing real-time training
and learning of hash-encoded neural SDFs. 

# Build

**Requirements**

- Windows 10/11 with Visual Studio 2022 or later
- CUDA 12+ and an NVIDIA GPU with compute capability $\geq$ 8.6 (RTX 30-series or newer for FullyFused MLP)
- CMake 3.22+ and Ninja

**Compile**

Open a Visual Studio developer command prompt and run:

```powershell
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

**Run a single configuration**

From `build/`:

```powershell
.\neural-sdf.exe ..\data\bunny.obj --run baseline --max_steps 3000
```

Output is written to `results/<mesh_stem>/`.
