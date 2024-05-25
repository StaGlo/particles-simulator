#ifndef PARTICLE_HPP
#define PARTICLE_HPP

// #include <iostream>

class Particle
{
private:
    double x, y, z;
    double vx, vy, vz;
    double mass;
    double radius;

public:
    __host__ __device__ Particle();
    __host__ __device__ Particle(double x, double y, double z, double vx, double vy, double vz, double mass, double radius);
    __host__ __device__ void updatePosition(double timestep);

    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }
};

// __global__ void updateSystem(Particle *particles, int numParticles, double timestep);

#endif // PARTICLE_HPP