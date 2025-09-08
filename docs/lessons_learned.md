# FEniCSx Lessons Learned

## Project Overview
This document captures the key lessons learned while setting up and working with FEniCSx v0.9.0 for finite element simulations, specifically solving the Poisson equation.

---

## 1. Environment Setup

### Python Environment
- **Use conda/mamba for FEniCSx installation**: Much more reliable than pip
- **Command that worked**:
  ```bash
  mamba create -n fenicsx-env -c conda-forge fenics-dolfinx mpich pyvista
  conda activate fenicsx-env
  ```
- **Key packages needed**: `fenics-dolfinx`, `mpich` (MPI implementation), `pyvista` (for visualization)

### WSL2 Integration
- **ParaView on Windows host can access WSL2 files** via `/mnt/c/` paths
- **File path format**: `/mnt/c/Program Files/ParaView 5.13.1/bin/paraview.exe`
- **WSL2 to Windows file access**: `\\wsl$\Ubuntu\home\username\`

---

## 2. DOLFINx API Understanding (v0.9.0)

### Critical API Discovery
DOLFINx v0.9.0 has **two parallel assembly APIs**:

1. **DOLFINx native objects** (for internal use):
   ```python
   A = fem.assemble_matrix(fem.form(a), bcs=[bc])  # Returns MatrixCSR
   b = fem.assemble_vector(fem.form(L))            # Returns Vector
   ```

2. **PETSc-compatible objects** (for solvers):
   ```python
   from dolfinx.fem import petsc as fem_petsc
   A = fem_petsc.assemble_matrix(fem.form(a), bcs=[bc])  # Returns PETSc.Mat
   b = fem_petsc.assemble_vector(fem.form(L))            # Returns PETSc.Vec
   ```

### Working Solution Pattern
```python
# Use fem_petsc for PETSc solvers
from dolfinx.fem import petsc as fem_petsc

A = fem_petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()  # PETSc matrices have .assemble() method

b = fem_petsc.assemble_vector(fem.form(L))
fem_petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem_petsc.set_bc(b, [bc])

# Solver setup
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.solve(b, uh.x.petsc_vec)  # Note: uh.x.petsc_vec not uh.vector
```

---

## 3. Debugging Strategy

### Systematic Diagnostic Approach
When facing API issues:

1. **Check what's actually available**:
   ```python
   print(dir(fem))  # See available functions
   print(hasattr(fem, 'petsc'))  # Check for submodules
   ```

2. **Inspect object types and methods**:
   ```python
   A = fem.assemble_matrix(fem.form(a), bcs=[bc])
   print(type(A))  # Know what you're working with
   print([m for m in dir(A) if not m.startswith('_')])  # Available methods
   ```

3. **Test incrementally**: Build up the solution step by step, testing each component

### ParaView Debugging
- **Use verbose logging**: `--verbosity 9 --enable-bt`
- **Check log files**: `--log paraview_debug.log,9`
- **Try alternative file formats** when one fails

---

## 4. File Format Compatibility

### XDMF vs VTK Issues
**Problem encountered**: XDMF files caused ParaView crashes
**Root cause**: Complex format with HDF5 dependencies
**Solution**: Use simpler VTK format for better compatibility

### Export Format Hierarchy (by reliability)
1. **Simple VTK** (.vtk) - Most compatible, ASCII format
2. **VTU/PVD** (.vtu, .pvd) - Modern VTK XML format
3. **VTX** (.bp) - New ADIOS2-based format
4. **XDMF** (.xdmf + .h5) - Complex but feature-rich

### Working Export Code
```python
# Multiple export formats for maximum compatibility
with io.XDMFFile(domain.comm, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

with io.VTXWriter(domain.comm, "solution.bp", [uh]) as vtx:
    vtx.write(0.0)

with io.VTKFile(domain.comm, "solution.pvd", "w") as vtk:
    vtk.write_function(uh)
```

---

## 5. Visualization Strategies

### Multi-layered Approach
1. **ParaView** (professional visualization)
   - Use simple VTK format for reliability
   - Access from Windows host in WSL2 environment

2. **Python matplotlib** (quick checks)
   - Always available, good for debugging
   - Create scatter plots, contours, 3D surfaces

3. **Raw data export** (backup/custom analysis)
   ```python
   # Export coordinates and values
   coords = V.tabulate_dof_coordinates()
   values = uh.x.array
   np.savez('solution.npz', coordinates=coords, values=values)
   ```

---

## 6. Project Organization Best Practices

### Directory Structure
```
FEniCSx/
├── helpers/           # Reusable utility scripts
│   ├── diagnose_xdmf.py
│   ├── simple_visualization.py
│   └── poisson_vtk_export.py
├── examples/          # Organized by topic
│   └── 01_poisson_basic/
│       └── poisson_demo.py
├── output/            # Generated files
│   ├── *.vtk, *.xdmf, *.png
│   └── *.log
└── docs/              # Documentation
    └── lessons_learned.md
```

### File Naming Convention
- **Scripts**: `descriptive_name.py`
- **Outputs**: `project_solution.format`
- **Logs**: `tool_debug.log`

---

## 7. Common Pitfalls and Solutions

### Problem: AttributeError: 'MatrixCSR' object has no attribute 'assemble'
**Cause**: Mixing DOLFINx native and PETSc APIs
**Solution**: Use `fem.petsc` consistently for PETSc solvers

### Problem: ParaView crashes on XDMF files
**Cause**: Complex format compatibility issues
**Solution**: Export to simple VTK format instead

### Problem: No verbose output from failing applications
**Cause**: Missing debug flags
**Solution**: Always use logging flags: `--verbosity 9 --enable-bt`

### Problem: File access issues between WSL2 and Windows
**Cause**: Path format differences
**Solution**: Copy files to Windows-accessible locations or use `\\wsl$\` paths

---

## 8. Performance Notes

### Solver Configuration
- **Direct solver**: `PETSc.KSP.Type.PREONLY` + `PETSc.PC.Type.LU`
  - Good for small problems (< 10,000 unknowns)
  - Exact solution

- **For larger problems**: Consider iterative solvers
  - `PETSc.KSP.Type.CG` + `PETSc.PC.Type.HYPRE`

### Memory Considerations
- **HDF5 files can be large** for fine meshes
- **VTK ASCII files are readable but larger** than binary
- **Use appropriate mesh resolution** for problem requirements

---

## 9. Testing and Validation

### Analytical Solution Verification
For Poisson equation ∇²u = -6 with BC u = 1 + x² + 2y²:
- **Expected solution**: u = 1 + x² + 2y²
- **Value range**: [1.0, 4.0] on unit square
- **Check**: Minimum at (0,0), Maximum at (1,1)

### Error Analysis
```python
# Compare numerical vs analytical
u_analytical = 1 + coords[:, 0]**2 + 2 * coords[:, 1]**2
error = np.abs(values - u_analytical)
max_error = np.max(error)
print(f"Maximum error: {max_error:.2e}")
```

---

## 10. Next Steps and Recommendations

### Immediate Improvements
1. **Create template scripts** for common PDE types
2. **Standardize error checking** and validation routines
3. **Develop mesh convergence studies** framework

### Advanced Topics to Explore
1. **Time-dependent problems** (heat equation)
2. **Nonlinear problems** (Navier-Stokes)
3. **Multi-physics coupling**
4. **Parallel computing** with MPI

### Tool Integration
1. **Jupyter notebooks** for interactive development
2. **Git version control** for code management
3. **Automated testing** for regression prevention

---

## Summary

The key to successful FEniCSx development is:
1. **Understand the dual API structure** (native vs PETSc)
2. **Use systematic debugging** when things fail
3. **Export multiple file formats** for visualization compatibility
4. **Organize code and outputs** systematically
5. **Always validate results** against known solutions

This foundation should enable productive finite element development with FEniCSx v0.9.0.