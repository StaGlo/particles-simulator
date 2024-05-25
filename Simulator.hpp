#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#define SPHERE_RADIUS 1000.0

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cuda_runtime.h>
#include "Particle.hpp"

class Simulator
{
public:
    Particle *h_particles;
    Particle *d_particles;

    int block_size, num_blocks;
    int num_particles, max_num_Particles;
    double timestep;
    unsigned long long int collisions = 0;

public:
    Simulator(double timestep, int num_particles);
    ~Simulator();

    // CUDA functions
    void allocateDeviceMemory();
    void copyToDevice();
    void copyFromDevice();
    void freeDeviceMemory();

    // Simulation functions
    void run(std::string filename, int numIterations);
    void addParticle(const Particle &p);
    void saveParticlePositions(std::string filename, int timestep);
    void updateSystem();

    // Getters
    unsigned long long int getCollisions() const { return collisions; }
    int getNumParticles() const { return num_particles; }

    // Setters
    void setBlockSize(int block_size) { this->block_size = block_size; }
    void setNumBlocks(int num_blocks) { this->num_blocks = num_blocks; }
};

#endif // SIMULATOR_HPP