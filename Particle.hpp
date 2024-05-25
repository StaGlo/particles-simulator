#ifndef PARTICLE_HPP
#define PARTICLE_HPP

class Particle
{
public:
    double x, y, z;
    double vx, vy, vz;
    double mass, radius;

    __host__ __device__ Particle();
    __host__ __device__ Particle(double x, double y, double z, double vx, double vy, double vz, double mass, double radius);

    __host__ __device__ void updatePosition(double timestep);

    // Getters
    __host__ __device__ double getX() const { return x; }
    __host__ __device__ double getY() const { return y; }
    __host__ __device__ double getZ() const { return z; }
    __host__ __device__ double getRadius() const { return radius; }
};

// Kernel function declaration
__global__ void updateSystemKernel(Particle *particles, int n, double timestep);

#endif // PARTICLE_HPP
