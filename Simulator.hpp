#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#define SPHERE_RADIUS 1000.0

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cuda_runtime.h>

struct Particle
{
    double x, y, z;
    double vx, vy, vz;
    double mass, radius;
};

class Simulator
{
public:
    Particle *h_particles;
    Particle *d_particles;

    int block_size, num_blocks;
    int num_particles, max_num_particles;
    double timestep;
    unsigned long long int collisions = 0;

public:
    Simulator(double timestep, int max_num_particles);
    ~Simulator();

    // Simulation functions
    void run(std::string filename, int numIterations);
    void addParticle(const Particle &p);
    void saveParticlePositions(std::string filename, int timestep);

    // Getters
    unsigned long long int getCollisions() const { return collisions; }
    int getNumParticles() const { return num_particles; }

    // Setters
    void setBlockSize(int block_size) { this->block_size = block_size; }
    void setNumBlocks(int num_blocks) { this->num_blocks = num_blocks; }
};

// Kernel function declaration
__global__ void updateSystemKernel(Particle *particles, int n, double timestep);

#endif // SIMULATOR_HPP