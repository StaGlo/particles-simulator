#include <omp.h>
#include <chrono>
#include <iostream>
#include "Particle.hpp"
#include "Simulator.hpp"

int main(int argc, char **argv)
{
    int num_threads;
    int particles;
    int iterations;

    if (argc == 1)
    {
        num_threads = 12;
        particles = 1000;
        iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iterations]" << std::endl;
        std::cout << "Using default number of threads: " << num_threads << std::endl;
        std::cout << "Using default number of particles: " << particles << std::endl;
        std::cout << "Using default number of iterations: " << iterations << std::endl;
    }

    else if (argc == 2)
    {
        num_threads = std::stoi(argv[1]);
        particles = 1000;
        iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iteartions]" << std::endl;
        std::cout << "Using default number of particles: " << particles << std::endl;
        std::cout << "Using default number of iterations: " << iterations << std::endl;
    }
    else if (argc == 3)
    {
        num_threads = std::stoi(argv[1]);
        particles = std::stoi(argv[2]);
        iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iterations]" << std::endl;
        std::cout << "Using default number of iterations: " << iterations << std::endl;
    }
    else if (argc == 4)
    {
        num_threads = std::stoi(argv[1]);
        particles = std::stoi(argv[2]);
        iterations = std::stoi(argv[3]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iterations]" << std::endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    Simulator sim(0.01);

    for (int i = 0; i < particles; i++)
    {
        Particle p(rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                   rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                   rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                   rand() % 1000 - 500,
                   rand() % 1000 - 500,
                   rand() % 1000 - 500,
                   rand() % 10 + 1,
                   rand() % 50);
        sim.addParticle(p);
    }

    auto start = std::chrono::high_resolution_clock::now();
    sim.run("particles.csv", iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Particles: " << sim.particles.size() << std::endl;
    std::cout << "Collisions: " << sim.get_collisions() << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
}
