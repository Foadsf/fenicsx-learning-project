#!/usr/bin/env python3
# transient_heat_conduction.py
# Solves the time-dependent heat equation (diffusion equation).
#
# Problem Description:
# A rectangular plate is initially at a uniform room temperature. A circular
# region in the center is suddenly heated and maintained at a high temperature,
# acting as a constant heat source. The outer edges of the plate are held
# at room temperature. We simulate how the temperature field evolves over time.
#
# PDE: ∂u/∂t = α ∇²u + f
#  - u: Temperature
#  - t: Time
#  - α: Thermal diffusivity (alpha = k / (rho * cp))
#  - f: Heat source/sink term (zero in this case)

import numpy as np
import ufl
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm  # For a progress bar


def create_problem_geometry():
    """Create a rectangular domain for the plate"""
    print("1. Creating problem geometry...")
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=[(-1.0, -1.0), (1.0, 1.0)],
        n=[50, 50],
        cell_type=mesh.CellType.triangle,
    )
    print(f"   ✓ Mesh created with {domain.topology.index_map(2).size_global} cells.")
    return domain


def define_physical_parameters():
    """Define simulation time and material properties for copper"""
    params = {
        # Simulation parameters
        "T_final": 10.0,  # Final time [s]
        "num_steps": 200,  # Number of time steps
        # Material properties (Copper)
        "k": 401.0,  # Thermal conductivity [W/(m·K)]
        "rho": 8960.0,  # Density [kg/m³]
        "cp": 385.0,  # Specific heat [J/(kg·K)]
        # Initial and boundary conditions
        "T_initial": 25.0,  # Initial temperature of the plate [°C]
        "T_boundary": 25.0,  # Temperature at the outer boundary [°C]
        "T_source": 300.0,  # Temperature of the central heat source [°C]
    }
    # Calculate thermal diffusivity
    params["alpha"] = params["k"] / (params["rho"] * params["cp"])
    # Calculate time step size
    params["dt"] = params["T_final"] / params["num_steps"]

    print("2. Defining physical parameters...")
    print(f"   ✓ Material: Copper (α = {params['alpha']:.2e} m²/s)")
    print(
        f"   ✓ Simulation time: {params['T_final']} s, Time step (dt): {params['dt']:.3f} s"
    )
    return params


def setup_time_dependent_problem(domain, params):
    """Setup the variational formulation for the transient heat equation"""
    print("3. Setting up transient variational problem...")

    # Create function space
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Define trial and test functions
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Define function to hold the solution from the previous time step
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.x.array[:] = params["T_initial"]

    # Define constants
    alpha = fem.Constant(domain, default_scalar_type(params["alpha"]))
    dt = fem.Constant(domain, default_scalar_type(params["dt"]))

    # Weak form using Backward Euler time discretization:
    # (u - u_n)/dt = α ∇²u
    # ∫(u - u_n)v dx = ∫ α dt ∇²u v dx
    # ∫uv dx - ∫u_n v dx = -∫ α dt ∇u ⋅ ∇v dx + ∫ α dt (∇u ⋅ n)v ds
    # Rearranging for LHS (unknown u) and RHS (known u_n):
    # ∫(u + α dt ∇u ⋅ ∇v)v dx = ∫u_n v dx
    F = (
        u * v * ufl.dx
        + alpha * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - u_n * v * ufl.dx
    )

    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))

    print("   ✓ Variational forms (a, L) created for Backward Euler.")
    return V, a, L, u_n


def define_boundary_conditions(domain, V, params):
    """Define time-dependent boundary conditions"""

    # 1. Outer boundary condition (fixed at T_boundary)
    # Since all exterior facets are at the same temperature, we can do this simply.
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_dofs = fem.locate_dofs_topological(
        V, domain.topology.dim - 1, mesh.exterior_facet_indices(domain.topology)
    )
    bc_outer = fem.dirichletbc(
        default_scalar_type(params["T_boundary"]), boundary_dofs, V
    )

    # 2. Inner circular heat source (fixed at T_source)
    def source_region(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2) < 0.2

    source_dofs = fem.locate_dofs_geometrical(V, source_region)
    bc_source = fem.dirichletbc(default_scalar_type(params["T_source"]), source_dofs, V)

    print("4. Defining boundary conditions...")
    print(f"   ✓ Outer walls fixed at {params['T_boundary']} °C")
    print(f"   ✓ Central heat source fixed at {params['T_source']} °C")

    return [bc_outer, bc_source]


def run_transient_simulation(domain, V, a, L, u_n, bcs, params):
    """Run the time-stepping loop to solve the transient problem"""

    print("\n--- Running Transient Simulation ---")

    # 1. Assemble the time-independent parts of the system
    A = fem_petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    # 2. Create solution function and output file
    uh = fem.Function(V)
    uh.name = "Temperature"

    output_path = "output/transient_heat_solution.xdmf"
    xdmf = io.XDMFFile(domain.comm, output_path, "w")
    xdmf.write_mesh(domain)

    # Write initial condition (t=0)
    uh.x.array[:] = u_n.x.array
    xdmf.write_function(uh, 0.0)

    # 3. Time-stepping loop
    t = 0.0
    progress = tqdm(range(params["num_steps"]), desc="Time-stepping")
    for i in progress:
        # Update time
        t += params["dt"]

        # Assemble the right-hand side vector (which depends on u_n)
        b = fem_petsc.assemble_vector(L)

        # Apply boundary conditions
        fem_petsc.apply_lifting(b, [a], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, bcs)

        # Setup solver
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        # Solve for the current time step
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        # Update the solution for the next time step
        u_n.x.array[:] = uh.x.array

        # Save solution to file at specified intervals
        if (i + 1) % 5 == 0:  # Save every 5 steps
            xdmf.write_function(uh, t)
            progress.set_postfix({"t": f"{t:.2f}s"})

    xdmf.close()
    print(f"   ✓ Simulation finished. Results saved to {output_path}")


def main():
    """Main function to run the transient heat analysis"""

    print("=" * 60)
    print("TRANSIENT HEAT CONDUCTION ANALYSIS")
    print("=" * 60)

    domain = create_problem_geometry()
    params = define_physical_parameters()
    V, a, L, u_n = setup_time_dependent_problem(domain, params)
    bcs = define_boundary_conditions(domain, V, params)
    run_transient_simulation(domain, V, a, L, u_n, bcs, params)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(
        "To visualize the time evolution, open '../../output/transient_heat_solution.xdmf' in ParaView."
    )
    print("Use the time controls in ParaView to play the animation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
