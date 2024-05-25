#include "Simulator.hpp"

Simulator::Simulator(double timestep, int max_num_Particles)
{
    this->timestep = timestep;
    this->max_num_Particles = max_num_Particles;
    num_particles = 0;
    h_particles = new Particle[max_num_Particles];
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
    if (std::filesystem::exists(filename))
    {
        std::filesystem::remove(filename);
    }

    allocateDeviceMemory();
    copyToDevice();
    std::cout << "Running simulation" << std::endl;
    for (int i = 0; i < numIterations; i++)
    {
        // updateSystem();
        updateSystemKernel<<<num_blocks, block_size>>>(d_particles, num_particles, timestep);
        cudaDeviceSynchronize();
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

__global__ void updateSystemKernel(Particle *particles, int num_particles, double timestep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles)
    {
        // particles[i].updatePosition(timestep);
        // Update position
        particles[i].x += particles[i].vx * timestep;
        particles[i].y += particles[i].vy * timestep;
        particles[i].z += particles[i].vz * timestep;
    }
}