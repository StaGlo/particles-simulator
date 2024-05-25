#include "Simulator.hpp"
#include <iostream>
#include <cuda_runtime.h>

Simulator::Simulator(double timestep, int maxNumParticles)
{
    this->timestep = timestep;
    this->numParticles = 0;
    this->maxNumParticles = maxNumParticles;
    this->h_particles = new Particle[maxNumParticles];
}

Simulator::~Simulator()
{
    delete[] h_particles;
    freeDeviceMemory();
}

void Simulator::allocateDeviceMemory()
{
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
}

void Simulator::copyToDevice()
{
    cudaMemcpy(d_particles, h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
}

void Simulator::copyFromDevice()
{
    cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
}

void Simulator::freeDeviceMemory()
{
    cudaFree(d_particles);
}

void Simulator::updateSystem()
{
    // TODO implement this function
}

void Simulator::addParticle(const Particle &p)
{
    h_particles[numParticles] = p;
    numParticles++;
}

void Simulator::run(std::string filename, int numIterations)
{
    allocateDeviceMemory();
    copyToDevice();
    for (int i = 0; i < numIterations; i++)
    {
        updateSystem();
        copyFromDevice();
        saveParticlePositions(filename, i);
    }
}

void Simulator::saveParticlePositions(std::string filename, int timestep)
{
    std::ofstream file;
    file.open(filename, std::ios::app);
    file << std::fixed << std::setprecision(2);
    for (int i = 0; i < numParticles; i++)
    {
        file << h_particles[i].getX() << " " << h_particles[i].getY() << " " << h_particles[i].getZ() << " ";
    }
    file << std::endl;
    file.close();
}