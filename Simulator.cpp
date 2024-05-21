#include "Simulator.hpp"

Simulator::Simulator(double timestep) : timestep(timestep) {}

void Simulator::updateSystem()
{
#pragma omp parallel for
    for (auto &particle : particles)
    {
        particle.updatePosition(timestep);
    }

    // Check for boundary collisions with the spherical boundary
#pragma omp parallel for
    for (auto &particle : particles)
    {
        double distanceFromCenter = std::sqrt(particle.x * particle.x + particle.y * particle.y + particle.z * particle.z);
        if (distanceFromCenter + particle.radius > SPHERE_RADIUS)
        {
            if (distanceFromCenter == 0)
            {
                std::cerr << "Error: Distance from center is zero." << std::endl;
                continue;
            }

            // Reflect velocity
            double normal_x = particle.x / distanceFromCenter;
            double normal_y = particle.y / distanceFromCenter;
            double normal_z = particle.z / distanceFromCenter;
            double dot_product = particle.vx * normal_x + particle.vy * normal_y + particle.vz * normal_z;
            particle.vx -= 2 * dot_product * normal_x;
            particle.vy -= 2 * dot_product * normal_y;
            particle.vz -= 2 * dot_product * normal_z;

            // Adjust position to be within the sphere
            double overlap = distanceFromCenter + particle.radius - SPHERE_RADIUS;
            particle.x -= overlap * normal_x;
            particle.y -= overlap * normal_y;
            particle.z -= overlap * normal_z;
        }
    }

    // Check for collisions between particles
#pragma omp parallel for collapse(2)
    for (int i = 0; i < particles.size(); ++i)
    {
        for (int j = i + 1; j < particles.size(); ++j)
        {

            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (distance <= particles[i].radius + particles[j].radius)
            {
                double m1 = particles[i].mass, m2 = particles[j].mass;
                double v1x_old = particles[i].vx, v1y_old = particles[i].vy, v1z_old = particles[i].vz;
                double v2x_old = particles[j].vx, v2y_old = particles[j].vy, v2z_old = particles[j].vz;

                // Compute new velocities using elastic collision formulas
                double v1x_new = (v1x_old * (m1 - m2) + 2 * m2 * v2x_old) / (m1 + m2);
                double v1y_new = (v1y_old * (m1 - m2) + 2 * m2 * v2y_old) / (m1 + m2);
                double v1z_new = (v1z_old * (m1 - m2) + 2 * m2 * v2z_old) / (m1 + m2);
                double v2x_new = (v2x_old * (m2 - m1) + 2 * m1 * v1x_old) / (m1 + m2);
                double v2y_new = (v2y_old * (m2 - m1) + 2 * m1 * v1y_old) / (m1 + m2);
                double v2z_new = (v2z_old * (m2 - m1) + 2 * m1 * v1z_old) / (m1 + m2);

#pragma omp critical
                {
                    particles[i].vx = v1x_new;
                    particles[i].vy = v1y_new;
                    particles[i].vz = v1z_new;
                    particles[j].vx = v2x_new;
                    particles[j].vy = v2y_new;
                    particles[j].vz = v2z_new;

                    collisions++;
                }
            }
        }
    }
}

void Simulator::addParticle(const Particle &p)
{
    if (p.x * p.x + p.y * p.y + p.z * p.z > SPHERE_RADIUS * SPHERE_RADIUS)
    {
        std::cerr << "Error: Particle is outside the sphere." << std::endl;
        return;
    }

    particles.push_back(p);
}

void Simulator::run(std::string filename, int numIterations)
{

    if (std::filesystem::exists(filename))
    {
        std::filesystem::remove(filename);
    }

    for (int i = 0; i < numIterations; i++)
    {
        updateSystem();
        saveParticlePositions(filename, i);
        // visualize();
    }
}

void Simulator::visualize()
{
    // Clear the screen
    std::cout << "\x1B[2J\x1B[H";

    // Create a grid
    std::vector<std::vector<char>> grid(this->height, std::vector<char>(this->width, ' '));

    // Draw particles
    for (int i = 0; i < this->particles.size(); i++)
    {
        auto particle = this->particles[i];
        int x = static_cast<int>((particle.x) / 1000 * this->width);
        int y = static_cast<int>((particle.y) / 1000 * this->height);

        if (x >= 0 && x < this->width && y >= 0 && y < this->height)
        {
            if (i == 0)
                grid[y][x] = '*';
            else
                grid[y][x] = 'O';
        }
    }

    // Draw box around particles
    for (int i = 0; i < this->width; i++)
    {
        grid[0][i] = '-';
        grid[this->height - 1][i] = '-';
    }
    for (int i = 0; i < this->height; i++)
    {
        grid[i][0] = '|';
        grid[i][this->width - 1] = '|';
    }

    // Print the grid
    for (int i = 0; i < this->height; i++)
    {
        for (int j = 0; j < this->width; j++)
        {
            std::cout << grid[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "Collisions: " << collisions << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void Simulator::saveParticlePositions(std::string filename, int timestepIndex)
{
    std::ofstream file(filename, std::ios_base::app);

    file << "Timestep" << timestepIndex << "\n";

    for (auto &particle : particles)
    {
        file << particle.x << "," << particle.y << "," << particle.z << "," << particle.radius << "\n";
    }

    file.close();
}