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
    std::vector<Particle> particles;
    double timestep;
    const int width = 100;
    const int height = 100;
    const int depth = 100;

    long collisions = 0;

public:
    Simulator(double timestep);

    void updateSystem();
    void addParticle(const Particle &p);
    void run(std::string filename, int numIterations);
    void visualize();
    long get_collisions() { return collisions; }
    void saveParticlePositions(std::string filename, int timestep);
};

#endif // SIMULATOR_HPP