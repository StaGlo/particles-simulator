# Particle Simulation Project

![simulation_gif](./particle_simulation.gif)

## Description
This project simulates the movement and collision of particles within a confined space. The simulation is implemented in both CPU and GPU versions to demonstrate the performance benefits of parallel computation using **OpenMP** and **CUDA**.

### Features
- Simulates particle motion in a 3D space.
- Detects and handles boundary collisions and collisions between particles.
- Programs written in **CPP**.
- Provides both CPU (multi-threaded with OpenMP) and GPU (CUDA) implementations.
- Outputs particle positions over time for visualization.

## Project Structure
The project contains the following key components:
- `openmp`- contains program files using OpenMP
- `cuda/`- contains CUDA version of the program

## How to run
To launch the simulation run:
- OpenMP:  
    - <code>g++ openmp/*.cpp -o run_openmp -fopenmp</code>
    - <code>./run_openmp</code>
- CUDA:
    - <code>nvcc cuda/*.cu -o run_cuda</code>
    - <code>./run_cuda</code>

To run the simulation's visualization run:  
- <code>python vizualize.py</code>