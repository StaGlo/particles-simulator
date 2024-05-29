#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "Simulator.hpp"

int main(int argc, char **argv)
{
    int num_particles;
    int num_iterations;

    if (argc == 1)
    {
        num_particles = 1000;
        num_iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_particles] [num_iterations]" << std::endl;
        std::cout << "Using default number of particles: " << num_particles << std::endl;
        std::cout << "Using default number of iterations: " << num_iterations << std::endl;
    }
    else if (argc == 2)
    {
        num_particles = std::stoi(argv[1]);
        num_iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_particles] [num_iterations]" << std::endl;
        std::cout << "Using default number of iterations: " << num_iterations << std::endl;
    }
    else if (argc == 3)
    {
        num_particles = std::stoi(argv[1]);
        num_iterations = std::stoi(argv[2]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [num_particles] [num_iterations]" << std::endl;
        return 1;
    }

    srand(0); // Constant seed for reproducibility
    Simulator sim(0.01, 1'000'000);

    for (int i = 0; i < num_particles; i++)
    {
        Particle p;
        p.x = rand() % int(SPHERE_RADIUS) - int(SPHERE_RADIUS) / 2;
        p.y = rand() % int(SPHERE_RADIUS) - int(SPHERE_RADIUS) / 2;
        p.z = rand() % int(SPHERE_RADIUS) - int(SPHERE_RADIUS) / 2;
        p.vx = rand() % 1001 - 500;
        p.vy = rand() % 1001 - 500;
        p.vz = rand() % 1001 - 500;
        p.mass = rand() % 10 + 1;
        p.radius = rand() % 50 + 1;

        sim.addParticle(p);
    }

    int block_size = 256;
    sim.setBlockSize(block_size);
    sim.setNumBlocks((num_particles + block_size - 1) / block_size);

    auto start = std::chrono::high_resolution_clock::now();
    sim.run("particles.csv", num_iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Particles: " << sim.getNumParticles() << std::endl;
    std::cout << "Collisions: " << sim.getCollisions() << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
}
