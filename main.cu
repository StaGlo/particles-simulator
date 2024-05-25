#include <chrono>
#include <iostream>
#include "Particle.hpp"
// #include "Simulator.hpp"

#define SPHERE_RADIUS 1000

int main(int argc, char **argv)
{
    int num_threads;
    int num_particles;
    int iterations;

    if (argc == 1)
    {
        num_threads = 12;
        num_particles = 1000;
        iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iterations]" << std::endl;
        std::cout << "Using default number of threads: " << num_threads << std::endl;
        std::cout << "Using default number of particles: " << num_particles << std::endl;
        std::cout << "Using default number of iterations: " << iterations << std::endl;
    }
    else if (argc == 2)
    {
        num_threads = std::stoi(argv[1]);
        num_particles = 1000;
        iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iteartions]" << std::endl;
        std::cout << "Using default number of particles: " << num_particles << std::endl;
        std::cout << "Using default number of iterations: " << iterations << std::endl;
    }
    else if (argc == 3)
    {
        num_threads = std::stoi(argv[1]);
        num_particles = std::stoi(argv[2]);
        iterations = 10'000;
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iterations]" << std::endl;
        std::cout << "Using default number of iterations: " << iterations << std::endl;
    }
    else if (argc == 4)
    {
        num_threads = std::stoi(argv[1]);
        num_particles = std::stoi(argv[2]);
        iterations = std::stoi(argv[3]);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [num_threads] [num_particles] [iterations]" << std::endl;
        return 1;
    }

    // Simulator sim(0.01);

    // srand(0); // Constant seed for reproducibility

    // TODO adjust code to use Simulator class
    for (int i = 0; i < num_particles; i++)
    {
        h_particles[i] = Particle(rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                                  rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                                  rand() % int(SPHERE_RADIUS) - SPHERE_RADIUS / 2,
                                  rand() % 1001 - 500,
                                  rand() % 1001 - 500,
                                  rand() % 1001 - 500,
                                  rand() % 10 + 1,
                                  rand() % 50 + 1);
        //  sim.addParticle(p);
    }

    Particle *d_particles;
    cudaMalloc(&d_particles, num_particles * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    auto start = std::chrono::high_resolution_clock::now();
    updateSystem<<<num_blocks, block_size>>>(d_particles, num_particles, 0.01);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(h_particles, d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    // auto start = std::chrono::high_resolution_clock::now();
    // sim.run("particles.csv", iterations);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;

    /* std::cout << "Threads: " << num_threads << std::endl;
     std::cout << "Particles: " << sim.particles.size() << std::endl;
     std::cout << "Collisions: " << sim.get_collisions() << std::endl*/
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
}
