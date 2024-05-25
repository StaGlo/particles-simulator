#include "Simulator.hpp"

Simulator::Simulator(double timestep, int max_num_Particles)
{
    this->timestep = timestep;
    this->max_num_Particles = max_num_Particles;
    num_particles = 0;
    h_particles = new Particle[max_num_Particles];

    block_size = 256;
    num_blocks = (num_particles + block_size - 1) / block_size;
}

Simulator::~Simulator()
{
    delete[] h_particles;
    freeDeviceMemory();
}

void Simulator::allocateDeviceMemory()
{
    cudaMalloc(&d_particles, num_particles * sizeof(Particle));
}

void Simulator::copyToDevice()
{
    cudaMemcpy(d_particles, h_particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
}

void Simulator::copyFromDevice()
{
    cudaMemcpy(h_particles, d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
}

void Simulator::freeDeviceMemory()
{
    cudaFree(d_particles);
}

void Simulator::addParticle(const Particle &p)
{
    h_particles[num_particles] = p;
    num_particles++;
}

void Simulator::updateSystem()
{
    updateSystemKernel<<<num_blocks, block_size>>>(d_particles, num_particles, timestep);
    cudaDeviceSynchronize();
}

void Simulator::run(std::string filename, int numIterations)
{
    allocateDeviceMemory();
    copyToDevice();
    for (int i = 0; i < numIterations; i++)
    {
        // updateSystemKernel<<<num_blocks, block_size>>>(d_particles, num_particles, timestep);
        updateSystem();
        cudaDeviceSynchronize();
        copyFromDevice();
        saveParticlePositions(filename, i);
    }
}

void Simulator::saveParticlePositions(std::string filename, int timestepIndex)
{
    std::ofstream file(filename, std::ios_base::app);

    file << "Timestep" << timestepIndex << "\n";

    for (int i = 0; i < num_particles; i++)
    {
        file << h_particles[i].getX() << "," << h_particles[i].getY() << "," << h_particles[i].getZ() << "," << h_particles[i].getRadius() << "\n";
    }

    file.close();
}