#include "Particle.hpp"

Particle::Particle(double x, double y, double z, double vx, double vy, double vz, double mass, double radius)
    : x(x), y(y), z(z), vx(vx), vy(vy), vz(vz), mass(mass), radius(radius)
{
}

void Particle::updatePosition(double timestep)
{
    x += vx * timestep;
    y += vy * timestep;
    z += vz * timestep;
}
