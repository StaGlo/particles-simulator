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
    double vx_new, vy_new, vz_new;
    double mass, radius;
};

class Simulator
{
private:
    Particle *h_particles;
    Particle *d_particles;
    unsigned long long int *d_collisions;
    unsigned long long int h_collisions;

    int block_size, num_blocks;
    int num_particles, max_num_particles;
    double timestep;

    void allocateDeviceMemory();
    void copyToDevice();
    void copyFromDevice();
    void freeDeviceMemory();
    
    void updateSystem();
    void saveParticlePositions(std::string filename, int timestep);

public:
    Simulator(double timestep, int max_num_particles);
    ~Simulator();

    // Simulation functions
    void run(std::string filename, int numIterations);
    void addParticle(const Particle &p);

    // Getters
    unsigned long long int getCollisions() const { return h_collisions; }
    int getNumParticles() const { return num_particles; }

    // Setters
    void setBlockSize(int block_size) { this->block_size = block_size; }
    void setNumBlocks(int num_blocks) { this->num_blocks = num_blocks; }
};

// Kernel function declaration
__global__ void updatePositionsCheckBoundaryCollisions(Particle *particles, int num_particles, double timestep);
__global__ void checkCollisions(Particle *particles, int num_particles, unsigned long long int *d_collisions);
__global__ void updateVelocities(Particle *particles, int num_particles);

#endif // SIMULATOR_HPP