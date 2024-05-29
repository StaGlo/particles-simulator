#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <iostream>

class Particle
{
public:
    double x, y, z;
    double vx, vy, vz;
    double mass;
    double radius;

    Particle(double x, double y, double z, double vx, double vy, double vz, double mass, double radius);

    void updatePosition(double timestep);
};

#endif // PARTICLE_HPP