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
    freeDeviceMemory();
}

void Simulator::allocateDeviceMemory()
{
    cudaMalloc(&d_particles, max_num_particles * sizeof(Particle));
    cudaMalloc(&d_collisions, sizeof(unsigned long long int));
}

void Simulator::copyToDevice()
{
    cudaMemcpy(d_particles, h_particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
}

void Simulator::copyFromDevice()
{
    cudaMemcpy(h_particles, d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_collisions, d_collisions, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    h_collisions /= 2; // Each collision is counted twice
}

void Simulator::freeDeviceMemory()
{
    cudaFree(d_particles);
    cudaFree(d_collisions);
}

void Simulator::addParticle(const Particle &p)
{
    h_particles[num_particles] = p;
    num_particles++;
}

void Simulator::updateSystem()
{
    // Update positions and check for boundary collisions
    updatePositionsCheckBoundaryCollisions<<<num_blocks, block_size>>>(d_particles, num_particles, timestep);
    cudaDeviceSynchronize();

    // Check for collisions between particles
    checkCollisions<<<num_blocks, block_size>>>(d_particles, num_particles, d_collisions);
    cudaDeviceSynchronize();

    // Update velocities
    updateVelocities<<<num_blocks, block_size>>>(d_particles, num_particles);
    cudaDeviceSynchronize();
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

void Simulator::run(std::string filename, int num_iterations)
{
    if (std::filesystem::exists(filename))
    {
        std::filesystem::remove(filename);
    }

    allocateDeviceMemory();
    copyToDevice();
    cudaMemset(d_collisions, 0, sizeof(unsigned long long int)); // Initialize collision counter

    for (int i = 0; i < num_iterations; i++)
    {
        updateSystem();
        copyFromDevice();
        saveParticlePositions(filename, i);
    }

    freeDeviceMemory();
}

__global__ void updatePositionsCheckBoundaryCollisions(Particle *particles, int num_particles, double timestep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles)
    {

        // Update position
        // printf("Updating position for particle %d\n", i);
        particles[i].x += particles[i].vx * timestep;
        particles[i].y += particles[i].vy * timestep;
        particles[i].z += particles[i].vz * timestep;

        // Check for collisions with walls
        // printf("Checking boundary collisions for particle %d\n", i);
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

__global__ void checkCollisions(Particle *particles, int num_particles, unsigned long long int *d_collisions)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles)
    {
        particles[i].vx_new = particles[i].vx;
        particles[i].vy_new = particles[i].vy;
        particles[i].vz_new = particles[i].vz;

        // printf("Checking collisions for particle %d\n", i);
        for (int j = 0; j < num_particles; j++)
        {
            if (i != j)
            {
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;
                double dz = particles[i].z - particles[j].z;
                double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (distance < particles[i].radius + particles[j].radius)
                {
                    atomicAdd(d_collisions, 1);

                    double mass_k = 1.0 / (particles[i].mass + particles[j].mass);
                    particles[i].vx_new = (particles[i].vx * (particles[i].mass - particles[j].mass) + 2 * particles[j].mass * particles[j].vx) * mass_k;
                    particles[i].vy_new = (particles[i].vy * (particles[i].mass - particles[j].mass) + 2 * particles[j].mass * particles[j].vy) * mass_k;
                    particles[i].vy_new = (particles[i].vz * (particles[i].mass - particles[j].mass) + 2 * particles[j].mass * particles[j].vz) * mass_k;
                }
            }
        }
    }
}

__global__ void updateVelocities(Particle *particles, int num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles)
    {
        // Update velocity
        // printf("Updating velocity for particle %d\n", i);
        particles[i].vx = particles[i].vx_new;
        particles[i].vy = particles[i].vy_new;
        particles[i].vz = particles[i].vz_new;
    }
}