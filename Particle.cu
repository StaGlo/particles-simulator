#include "Particle.hpp"

__host__ __device__ Particle::Particle()
    : x(0), y(0), z(0), vx(0), vy(0), vz(0) {}

__host__ __device__ Particle::Particle(double x, double y, double z, double vx, double vy, double vz, double mass, double radius)
    : x(x), y(y), z(z), vx(vx), vy(vy), vz(vz), mass(mass), radius(radius) {}

__host__ __device__ void Particle::updatePosition(double timestep)
{
    x += vx * timestep;
    y += vy * timestep;
    z += vz * timestep;
}

__global__ void updateSystemKernel(Particle *particles, int num_particles, double timestep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles)
    {
        particles[i].updatePosition(timestep);
    }
}