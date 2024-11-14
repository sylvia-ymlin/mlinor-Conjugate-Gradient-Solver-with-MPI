# MPI Parallel Conjugate Gradient Solver

## Overview
This code is an implementation of the Conjugate Gradient (CG) method based on MPI. It is designed to solve a 2D Poisson equation with Dirichlet boundary conditions.

## Files
- `CG.c`: The main source file containing the MPI parallel implementation of the CG method.
- `README.md`: This file providing an overview of the project.
- `Makefile`: Makefile for compiling the code.

## Environment Configuration

The code works on on UPPMAX snowy. Make sure the following modules are loaded.

```bash
gcc/12.2.0 
openmpi/4.1.4
```

## Compilation
To compile the code, use the provided Makefile:
```bash
make
```

Removes all binary files and object files:

```bash
make clean
```

## Running the Code

Use the command bellow to run the code

```bash
mpirun --bind-to none -n {p} ./CG {n}
```

`p`: the number of processes used. It should be a perfect square number.

`n`: the input parameter, the number of intervals along each coordinate axis.

## Output

The output of the code is the residual defined by
$$
\|Au-b\|_2
$$
, printed to the console.