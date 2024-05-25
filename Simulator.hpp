#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#define SPHERE_RADIUS 1000.0

#include <vector>
#include <iostream>
#include <thread>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include "Particle.hpp"

class Simulator
{
public:
    Particle *h_particles;
    Particle *d_particles;

    int numParticles, maxNumParticles;
    double timestep;
    unsigned long long int collisions = 0;

public:
    Simulator(double timestep, int numParticles);
    ~Simulator();

    // CUDA functions
    void allocateDeviceMemory();
    void copyToDevice();
    void copyFromDevice();
    void freeDeviceMemory();

    void updateSystem();
    void addParticle(const Particle &p);
    void run(std::string filename, int numIterations);
    long get_collisions() { return collisions; }
    void saveParticlePositions(std::string filename, int timestep);
};

#endif // SIMULATOR_HPP