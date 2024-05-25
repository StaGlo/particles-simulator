#include <chrono>
#include <iostream>
#include "Particle.hpp"
#include "Simulator.hpp"

int main(int argc, char **argv)
{
    int num_threads;
    int num_particles;
    int num_iterations;

    if (argc == 1)
    {
        num_threads = 12;
        num_particles = 1000;
        num_iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [num_iterations]" << std::endl;
        std::cout << "Using default number of threads: " << num_threads << std::endl;
        std::cout << "Using default number of particles: " << num_particles << std::endl;
        std::cout << "Using default number of iterations: " << num_iterations << std::endl;
    }
    else if (argc == 2)
    {
        num_threads = std::stoi(argv[1]);
        num_particles = 1000;
        num_iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [num_iterations]" << std::endl;
        std::cout << "Using default number of particles: " << num_particles << std::endl;
        std::cout << "Using default number of iterations: " << num_iterations << std::endl;
    }
    else if (argc == 3)
    {
        num_threads = std::stoi(argv[1]);
        num_particles = std::stoi(argv[2]);
        num_iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [num_iterations]" << std::endl;
        std::cout << "Using default number of iterations: " << num_iterations << std::endl;
    }
    else if (argc == 4)
    {
        num_threads = std::stoi(argv[1]);
        num_particles = std::stoi(argv[2]);
        num_iterations = std::stoi(argv[3]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [num_iterations]" << std::endl;
        return 1;
    }

    Simulator sim(0.01, 1'000'000);
    srand(0); // Constant seed for reproducibility

    for (int i = 0; i < num_particles; i++)
    {
        Particle p = Particle(rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                              rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                              rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                              rand() % 1001 - 500,
                              rand() % 1001 - 500,
                              rand() % 1001 - 500,
                              rand() % 10 + 1,
                              rand() % 50 + 1);
        sim.addParticle(p);
    }

        auto start = std::chrono::high_resolution_clock::now();
    sim.run("particles.csv", num_iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Particles: " << sim.getNumParticles() << std::endl;
    std::cout << "Collisions: " << sim.getCollisions() << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
}
