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

---

## 11. File Format Compatibility and Diagnostics

### ParaView Compatibility Issues
**Problem discovered**: XDMF and ADIOS2 BP formats often fail to open in Windows ParaView
**Root causes identified**:
- WSL2/Windows path compatibility issues
- Different HDF5 library versions between FEniCSx and ParaView
- ADIOS2 format not consistently supported across ParaView builds

**Solutions developed**:
1. **Multi-format export strategy**: Always export to VTK as fallback
2. **Diagnostic tools**: Created comprehensive file health checker
3. **Format-specific fixes**: Different approaches for each file type

### File Health Checking
**Created diagnostic workflow**:
```python
# Check individual files with automatic visualization
python3 helpers/check_file_health.py output/file.vtk
python3 helpers/check_file_health.py output/data.bp

# Comprehensive directory scan
python3 helpers/check_file_health.py output/
```

**Diagnostic capabilities developed**:
- **VTK validation**: Header parsing, data section verification
- **XDMF validation**: XML structure, HDF5 reference checking
- **BP format analysis**: ADIOS2 metadata inspection and variable reading
- **Automatic visualization**: Data plotting for healthy files

---

## 12. Advanced Boundary Conditions Implementation

### Mixed Boundary Conditions
**Challenge**: Implementing realistic engineering boundary conditions
**Achieved in heat conduction example**:
- **Dirichlet BC**: `fem.dirichletbc(T_hot, left_dofs, V)`
- **Robin BC**: Surface integral `h * T * v * ds(boundary_id)`
- **Neumann BC**: Natural boundary condition (no explicit implementation needed)

**Key implementation details**:
```python
# Boundary marking
right_facets = mesh.locate_entities_boundary(domain, 1, right_boundary)
marked_values = np.full(len(right_facets), 1, dtype=np.int32)
facet_tag = mesh.meshtags(domain, 1, right_facets, marked_values)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# Robin BC in weak form
a_convection = h * T * v * ds(1)  # Left-hand side
L_convection = h * T_air * v * ds(1)  # Right-hand side
```

---

## 13. Coupled Physics Problems

### Thermal Stress Analysis
**Challenge**: Coupling thermal and mechanical physics
**Implementation approach**:
1. **Sequential coupling**: Solve thermal problem first
2. **Function space interpolation**: Transfer temperature to mechanical problem
3. **Simplified thermal loading**: Equivalent body forces instead of full thermal strain

**Key lessons**:
- **Function space compatibility**: Direct coupling causes UFL compilation errors
- **Interpolation necessity**: `T_mech.interpolate(T_h)` required for cross-physics data transfer
- **API complexity**: Full thermal strain implementation requires advanced UFL techniques

**Working solution pattern**:
```python
# Solve thermal problem
T_h = solve_thermal_problem(domain, props)

# Transfer to mechanical problem
V_T_mech = fem.functionspace(domain, ("Lagrange", 1))
T_mech = fem.Function(V_T_mech)
T_mech.interpolate(T_h)

# Use in mechanical analysis
thermal_force = alpha * E * (T_avg - T_ref) / domain.geometry.dim
```

---

## 14. ADIOS2 Integration and BP File Handling

### ADIOS2 Python API Discovery
**Challenge**: Limited documentation for ADIOS2 Python bindings
**API patterns discovered**:
```python
# Correct ADIOS2 usage
reader = adios2.FileReader(str(bp_directory))
variables = reader.available_variables()
data = reader.read(variable_name)
reader.close()
```

**Common API mistakes to avoid**:
- `adios2.open()` doesn't exist (use `FileReader`)
- `adios2.ADIOS()` vs `adios2.Adios()` (case sensitivity)
- Mode constants are integers, not enums
- All methods use snake_case, not PascalCase

### BP File Structure Understanding
**ADIOS2 BP format consists of**:
- `data.0`: Binary simulation data
- `md.0`, `md.idx`: Metadata files
- `mmd.0`: Additional metadata
- `profiling.json`: Performance information

**Health indicators**:
- All required files present
- Non-zero metadata file size (>100 bytes typically)
- Readable variable list with expected names

---

## 15. Environment Management and Cross-Platform Issues

### WSL2 + Windows Integration
**File path handling**:
- WSL2 paths: `/mnt/c/dev/project/`
- Windows paths: `C:\dev\project\`
- ParaView access: `"C:\Program Files\ParaView\bin\paraview.exe" file.vtk`

**Python environment separation**:
- **WSL2**: FEniCSx conda environment for simulation
- **Windows**: Separate Python for Windows-only tools
- **Recommendation**: Run all analysis from WSL2 environment

**Key integration commands**:
```bash
# From WSL2, open Windows ParaView
"/mnt/c/Program Files/ParaView 5.13.1/bin/paraview.exe" output/file.vtk

# Access WSL2 files from Windows
\\wsl$\Ubuntu\home\username\project\
```

---

## 16. Performance and Optimization Notes

### File Format Performance
**Format comparison for medium problems (1000-10000 DOF)**:
- **VTK ASCII**: Slow write, universal compatibility
- **XDMF+HDF5**: Fast write, compatibility issues
- **ADIOS2 BP**: Fastest write, visualization challenges
- **Raw NumPy**: Fastest for post-processing

**Memory usage observations**:
- **PETSc solvers**: Efficient for sparse systems
- **Function space interpolation**: Memory overhead for cross-physics coupling
- **Visualization data**: Store only essential fields to reduce file sizes

### Solver Configuration
**Tested configurations**:
- **Small problems (<5000 DOF)**: `PREONLY` + `LU` (direct solver)
- **Thermal problems**: Generally well-conditioned, fast convergence
- **Coupled problems**: Sequential solving more robust than monolithic

---

## 17. Quality Assurance and Validation

### Solution Verification Strategy
**Multi-level validation implemented**:
1. **Analytical comparison**: Known exact solutions where possible
2. **Physical bounds checking**: Temperature/stress ranges
3. **Conservation verification**: Energy balance for thermal problems
4. **Mesh convergence**: Systematic refinement studies

**Automated checks in examples**:
```python
# Range validation
T_min, T_max = T_h.x.array.min(), T_h.x.array.max()
assert T_min >= props["T_air"], "Temperature below ambient"
assert T_max <= props["T_hot"] + 1e-6, "Temperature above boundary condition"

# Conservation check
total_heat_generated = props['q_source'] * domain_area
```

### Regression Testing Framework
**File health checker as testing tool**:
- Automatically validates all output formats
- Detects file corruption or format changes
- Provides quantitative metrics for solution comparison
- Enables automated CI/CD integration

---

## 18. Nonlinear PDE Solving in DOLFINx v0.9.0

### Newton Solver Implementation
**Challenge**: Solving nonlinear PDEs where coefficients depend on the solution
**Example problem**: -∇·((1+u²)∇u) = f

**DOLFINx v0.9.0 nonlinear solver API**:
```python
from dolfinx.nls.petsc import NewtonSolver

# Define residual form F(u;v) = 0
F = ufl.inner((1 + uh**2) * ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

# Setup nonlinear problem
problem = fem_petsc.NonlinearProblem(F, uh, bcs=[bc])
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# Configure and solve
solver.rtol = 1e-8
solver.max_it = 20
n, converged = solver.solve(uh)
```

**Key API locations discovered**:
- Newton solver: `dolfinx.nls.petsc.NewtonSolver` (not in `fem.petsc`)
- Nonlinear problem: `fem_petsc.NonlinearProblem`
- No `errornorm` function available - compute manually

### Manufactured Solution Workflow
**Using SymPy for solution verification**:
```python
import sympy

# Define analytical solution
u_exact = 1 + x + 2*y + sympy.sin(sympy.pi*x)*sympy.cos(sympy.pi*y)

# Compute source term by substitution
k_expr = 1 + u_exact**2
f_expr = sympy.diff(-k_expr * sympy.diff(u_exact, x), x) + \
         sympy.diff(-k_expr * sympy.diff(u_exact, y), y)

# Convert to C code for DOLFINx
u_code = sympy.printing.ccode(u_exact_expr)
f_code = sympy.printing.ccode(f_expr)
```

**Benefits of manufactured solutions**:
- Exact error quantification
- Verification of nonlinear solver implementation
- Code validation independent of physical interpretation

### High-Order Elements and Export Issues
**Challenge**: P2 elements incompatible with XDMF mesh degree
**Error**: "Degree of output Function must be same as mesh degree"

**Solution**: Interpolate to P1 for export
```python
# Solve on P2 space
V_solve = fem.functionspace(domain, ("Lagrange", 2))
uh = fem.Function(V_solve)  # P2 solution

# Export on P1 space
V_export = fem.functionspace(domain, ("Lagrange", 1))
uh_export = fem.Function(V_export)
uh_export.interpolate(uh)
```

**Alternative approaches**:
- Use VTK format which handles higher-order elements better
- Export raw data for custom visualization
- Use native DOLFINx visualization functions

### Error Computation Without errornorm
**Manual L2 error calculation**:
```python
# L2 norm: ||u_h - u_exact||_L2 = sqrt(∫(u_h - u_exact)² dx)
error_expr = (uh - u_exact)**2
error_L2_squared = fem.assemble_scalar(fem.form(error_expr * ufl.dx))
error_L2 = np.sqrt(error_L2_squared)

# Pointwise maximum error
error_max = np.max(np.abs(uh.x.array - u_exact.x.array))
```

### Performance Characteristics
**Observed Newton convergence**:
- **Typical iterations**: 5-15 for well-posed problems
- **Convergence rate**: Quadratic near solution
- **Tolerance achievable**: 1e-8 to 1e-12 for smooth problems

**Element choice impact**:
- **P1 elements**: Fast assembly, lower accuracy
- **P2 elements**: Better accuracy for smooth solutions, moderate cost
- **P3+ elements**: Diminishing returns unless solution is very smooth

### Mesh Topology Requirements
**Critical setup step**:
```python
# Required before boundary condition setup
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
```

**Common error without connectivity**:
"Facet to cell connectivity has not been computed"

**Best practice**: Always compute required connectivity immediately after mesh creation

### Nonlinear Problem Robustness
**Convergence factors**:
- **Initial guess quality**: Good initial approximation crucial
- **Mesh resolution**: Sufficient resolution for capturing nonlinearity
- **Boundary conditions**: Well-posed problem formulation
- **Solver tolerances**: Balance accuracy vs. computational cost

**Debugging nonlinear convergence**:
- Check residual norm progression
- Verify manufactured solution setup
- Test with simpler linear problem first
- Use continuation methods for difficult problems

---

## Summary

The key to successful FEniCSx development is:
1. **Understand the dual API structure** (native vs PETSc)
2. **Use systematic debugging** when things fail
3. **Export multiple file formats** for visualization compatibility
4. **Organize code and outputs** systematically
5. **Always validate results** against known solutions

This foundation should enable productive finite element development with FEniCSx v0.9.0.
