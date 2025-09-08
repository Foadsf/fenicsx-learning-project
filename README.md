# FEniCSx Learning Project

## Overview
This repository contains learning materials, examples, and utilities for working with FEniCSx v0.9.0 finite element software.

## Directory Structure

```
FEniCSx/
├── README.md              # This file
├── docs/
│   └── lessons_learned.md  # Detailed lessons learned
├── helpers/                # Utility scripts
│   ├── diagnose_xdmf.py    # XDMF file diagnostics
│   ├── simple_visualization.py  # Python-based visualization
│   ├── visualize_solution.py    # Multi-method visualization
│   └── poisson_vtk_export.py    # Multi-format export utility
├── examples/               # Organized by complexity
│   └── 01_poisson_basic/
│       └── poisson_demo.py # Basic Poisson equation solver
└── output/                 # Generated files (.vtk, .xdmf, .png, etc.)
```

## Quick Start

### Environment Setup
```bash
# Create conda environment
mamba create -n fenicsx-env -c conda-forge fenics-dolfinx mpich pyvista
conda activate fenicsx-env

# Additional packages for visualization
pip install h5py matplotlib
```

### Running Examples
```bash
cd examples/01_poisson_basic
python3 poisson_demo.py
```

### Visualization
```bash
# Python-based visualization
cd ../../helpers
python3 simple_visualization.py

# ParaView (from WSL2 with Windows ParaView)
"/mnt/c/Program Files/ParaView 5.13.1/bin/paraview.exe" ../output/poisson_simple.vtk
```

## Key Files

### Examples
- **`examples/01_poisson_basic/poisson_demo.py`**: Complete Poisson equation solver
  - Solves ∇²u = -6 on unit square
  - Dirichlet BC: u = 1 + x² + 2y²
  - Uses DOLFINx v0.9.0 PETSc API

### Utilities
- **`helpers/diagnose_xdmf.py`**: Diagnose XDMF/HDF5 file issues
- **`helpers/simple_visualization.py`**: Create plots without ParaView
- **`helpers/poisson_vtk_export.py`**: Export solutions in multiple formats

### Documentation
- **`docs/lessons_learned.md`**: Comprehensive lessons learned document
- **`README.md`**: This overview file

## Common Tasks

### Solve and Visualize PDE
```bash
# 1. Solve the problem
cd examples/01_poisson_basic
python3 poisson_demo.py

# 2. Create multiple export formats
cd ../../helpers
python3 poisson_vtk_export.py

# 3. Generate Python visualization
python3 simple_visualization.py

# 4. Open in ParaView
"/mnt/c/Program Files/ParaView 5.13.1/bin/paraview.exe" ../output/poisson_simple.vtk
```

### Debug File Format Issues
```bash
cd helpers
python3 diagnose_xdmf.py
```

## Environment Notes

### WSL2 + Windows ParaView
- **File access**: Use `/mnt/c/` paths or `\\wsl$\Ubuntu\` from Windows
- **ParaView command**: Full Windows path with quotes for spaces
- **File format**: Simple VTK (.vtk) most reliable, XDMF can be problematic

### FEniCSx v0.9.0 API
- **Use `dolfinx.fem.petsc` for PETSc solvers**
- **Matrix assembly**: `fem_petsc.assemble_matrix()` returns `PETSc.Mat`
- **Vector operations**: Use `uh.x.petsc_vec` for solver interface

## Troubleshooting

### Common Issues
1. **"MatrixCSR has no attribute 'assemble'"**
   - Solution: Use `fem_petsc.assemble_matrix()` instead of `fem.assemble_matrix()`

2. **ParaView crashes on file open**
   - Solution: Try simple VTK format instead of XDMF
   - Use: `python3 poisson_vtk_export.py` to create compatible formats

3. **Module not found errors**
   - Solution: Install missing packages: `pip install h5py matplotlib`

### Getting Help
- Check `docs/lessons_learned.md` for detailed troubleshooting
- Use diagnostic scripts in `helpers/` folder
- Ensure all dependencies are installed in conda environment

## Next Steps
- Explore time-dependent problems
- Try nonlinear PDEs
- Learn parallel computing with MPI
- Study mesh adaptivity techniques

---

*Generated: September 8, 2025*  
*FEniCSx Version: 0.9.0*  
*Environment: WSL2 Ubuntu + Windows ParaView*