#include "Simulator.hpp"

Simulator::Simulator(double timestep, int max_num_Particles)
{
    this->timestep = timestep;
    this->max_num_particles = max_num_Particles;
    num_particles = 0;
    h_particles = new Particle[this->max_num_particles];
}

Simulator::~Simulator()
{
    delete[] h_particles;
    cudaFree(d_particles);
}

void Simulator::addParticle(const Particle &p)
{
    h_particles[num_particles] = p;
    num_particles++;
}

void Simulator::run(std::string filename, int num_iterations)
{
    if (std::filesystem::exists(filename))
    {
        std::filesystem::remove(filename);
    }

    cudaMalloc(&d_particles, num_particles * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    for (int i = 0; i < num_iterations; i++)
    {
        updateSystemKernel<<<num_blocks, block_size>>>(d_particles, num_particles, timestep);
        cudaDeviceSynchronize();
        cudaMemcpy(h_particles, d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
        saveParticlePositions(filename, i);
    }

    cudaFree(d_particles);
}

void Simulator::saveParticlePositions(std::string filename, int timestepIndex)
{
    std::ofstream file(filename, std::ios_base::app);

    file << "Timestep" << timestepIndex << "\n";

    for (int i = 0; i < num_particles; i++)
    {
        file << h_particles[i].x << "," << h_particles[i].y << "," << h_particles[i].z << "," << h_particles[i].radius << "\n";
    }

    file.close();
}

__global__ void updateSystemKernel(Particle *particles, int num_particles, double timestep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles)
    {
        // Update position
        particles[i].x += particles[i].vx * timestep;
        particles[i].y += particles[i].vy * timestep;
        particles[i].z += particles[i].vz * timestep;

        // Check for collisions with walls
        double distanceFromCenter = std::sqrt(particles[i].x * particles[i].x + particles[i].y * particles[i].y + particles[i].z * particles[i].z);
        if (distanceFromCenter + particles[i].radius > SPHERE_RADIUS)
        {
            if (distanceFromCenter == 0)
            {
                printf("Error: Distance from center is zero.\n");
                return;
            }

            // Reflect velocity
            double normal_x = particles[i].x / distanceFromCenter;
            double normal_y = particles[i].y / distanceFromCenter;
            double normal_z = particles[i].z / distanceFromCenter;
            double dot_product = particles[i].vx * normal_x + particles[i].vy * normal_y + particles[i].vz * normal_z;
            particles[i].vx -= 2 * dot_product * normal_x;
            particles[i].vy -= 2 * dot_product * normal_y;
            particles[i].vz -= 2 * dot_product * normal_z;

            // Adjust position to be within the sphere
            double overlap = distanceFromCenter + particles[i].radius - SPHERE_RADIUS;
            particles[i].x -= overlap * normal_x;
            particles[i].y -= overlap * normal_y;
            particles[i].z -= overlap * normal_z;
        }
    }
}
