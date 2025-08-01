#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <ctime>

namespace fs = std::filesystem;

// macro: constant parameters
#define NUM_VARS 4  // number of independent variables
#define nGhost 2  // number of ghost cells in each direction
#define C 0.8  // CFL number
#define gamma 1.4  // adiabatic index of ideal gas
#define nThreadsX 32  // number of threads per block in x-direction
#define nThreadsY 32  // number of threads per block in y-direction
#define nThreadsXSLICX 32  // number of threads per block in x-direction for SLIC evolution in x-direction
#define nThreadsYSLICX 4  // number of threads per block in y-direction for SLIC evolution in x-direction
#define nThreadsXSLICY 4  // number of threads per block in x-direction for SLIC evolution in y-direction
#define nThreadsYSLICY 32  // number of threads per block in y-direction for SLIC evolution in y-direction

// macro: debugging
#define CUDA_CHECK {\
    cudaDeviceSynchronize();\
    cudaError_t err = cudaGetLastError();\
    if(err){\
        std::cout << "Error: " << cudaGetErrorString(err) << " line " << __LINE__ << std::endl;\
        exit(1);\
    }\
}

enum Processor {CPU, GPU};

// structure: data storage and process
struct Grid {

    // attributes
    double* data;
    int nCellsX;
    int nCellsY;
    Processor processor;

    // format for building an instance
    Grid(const int nX, const int nY, Processor p) {
        nCellsX = nX;
        nCellsY = nY;
        processor = p;
        switch (processor) {
            case CPU:
                data = new double[(nCellsX + 2 * nGhost) * (nCellsY + 2 * nGhost) * NUM_VARS];
                break;
            case GPU:
                cudaMalloc((void **)& data, (nCellsX + 2 * nGhost) * (nCellsY + 2 * nGhost) * NUM_VARS * sizeof(double));
                break;
        }
    }

    // operator functions
    __device__ __host__
    double operator() (const int i, const int j, const int v) const {
        return data[i + j * (nCellsX + 2 * nGhost) + v * (nCellsX + 2 * nGhost) * (nCellsY + 2 * nGhost)];
    }
    __device__ __host__
    double& operator() (const int i, const int j, const int v) {
        return data[i + j * (nCellsX + 2 * nGhost) + v * (nCellsX + 2 * nGhost) * (nCellsY + 2 * nGhost)];
    }
};


// function: transform from primitive to conservative on CPU
void prim2consHost(double* u_ij_cons, const double* u_ij_prim) {

    const double rho = u_ij_prim[0];
    const double u = u_ij_prim[1];
    const double v = u_ij_prim[2];
    const double p = u_ij_prim[3];

    u_ij_cons[0] = rho;  // rho
    u_ij_cons[1] = rho * u;  // momx
    u_ij_cons[2] = rho * v;  // momy
    u_ij_cons[3] = p / (gamma - 1) + 0.5 * rho * (pow(u, 2) + pow(v, 2));  // E
}


// function: transform from primitive to conservative on GPU
__device__ void prim2consDevice(double* u_ij_cons, const double* u_ij_prim) {

    const double rho = u_ij_prim[0];
    const double u = u_ij_prim[1];
    const double v = u_ij_prim[2];
    const double p = u_ij_prim[3];

    u_ij_cons[0] = rho;  // rho
    u_ij_cons[1] = rho * u;  // momx
    u_ij_cons[2] = rho * v;  // momy
    u_ij_cons[3] = p / (gamma - 1) + 0.5 * rho * (pow(u, 2) + pow(v, 2));  // E
}


// function: transform from conservative to primitive on CPU
void cons2primHost(double* u_ij_prim, const double* u_ij_cons) {

    const double rho = u_ij_cons[0];
    const double momx = u_ij_cons[1];
    const double momy = u_ij_cons[2];
    const double E = u_ij_cons[3];

    u_ij_prim[0] = rho;  // rho
    u_ij_prim[1] = momx / rho;  // u
    u_ij_prim[2] = momy / rho;  // v
    u_ij_prim[3] = (gamma - 1) * (E - 0.5 * pow(momx, 2) / rho - 0.5 * pow(momy, 2) / rho);  // p
}


// function: transform from conservative to primitive on GPU
__device__ void cons2primDevice(double* u_ij_prim, const double* u_ij_cons) {

    const double rho = u_ij_cons[0];
    const double momx = u_ij_cons[1];
    const double momy = u_ij_cons[2];
    const double E = u_ij_cons[3];

    u_ij_prim[0] = rho;  // rho
    u_ij_prim[1] = momx / rho;  // u
    u_ij_prim[2] = momy / rho;  // v
    u_ij_prim[3] = (gamma - 1) * (E - 0.5 * pow(momx, 2) / rho - 0.5 * pow(momy, 2) / rho);  // p
}


// function: set transmissive boundary conditions
__global__ void setBoundaryCondition(Grid u) {

    const int nCellsX = u.nCellsX, nCellsY = u.nCellsY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // transmissive boundary condition
    for (int v = 0; v < NUM_VARS; v++) {
        // lower boundary
        if (i >= nGhost && i < nCellsX + nGhost && j >= 0 && j < nGhost) {
            u(i, j, v) = u(i, nGhost, v);
        }
        // upper boundary
        if (i >= nGhost && i < nCellsX + nGhost && j >= nCellsY + nGhost && j < nCellsY + 2 * nGhost) {
            u(i, j, v) = u(i, nCellsY + nGhost - 1, v);
        }
        // left boundary
        if (j >= 0 && j < nCellsY + 2 * nGhost && i >= 0 && i < nGhost) {
            u(i, j, v) = u(nGhost, j, v);
        }
        // right boundary
        if (j >= 0 && j < nCellsY + 2 * nGhost && i >= nCellsX + nGhost && i < nCellsX + 2 * nGhost) {
            u(i, j, v) = u(nCellsX + nGhost - 1, j, v);
        }
    }
}


// function: calculate the maximum velocity in each block on GPU
__global__ void computeAmaxOpt(double* aDevice, Grid u) {

    // shared memory for current block
    __shared__ double aBlock[nThreadsX][nThreadsY];

    // variable substitution
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_local = threadIdx.x, j_local = threadIdx.y;
    double cur_a = 0.0;

    // calculate a's in current block
    if (i >= nGhost && i < u.nCellsX + nGhost && j >= nGhost && j < u.nCellsY + nGhost) {
        double u_prim[NUM_VARS];
        double u_cons[NUM_VARS];
        for (int v = 0; v < NUM_VARS; v++) {
            u_cons[v] = u(i, j, v);
        }
        cons2primDevice(u_prim, u_cons);
        double cur_rho = u_prim[0], cur_vx = u_prim[1], cur_vy = u_prim[2], cur_p = u_prim[3];
        double cur_v = pow(pow(cur_vx, 2) + pow(cur_vy, 2), 0.5);
        double Cs = pow(gamma * cur_p / cur_rho, 0.5);
        cur_a = cur_v + Cs;
    }
    aBlock[threadIdx.x][threadIdx.y] = cur_a;
    __syncthreads();

    // block-wise reduction
    if (blockDim.x >= 32 && blockDim.y >= 32 && i_local < 16 && j_local < 16) {
        aBlock[i_local][j_local] = fmax(fmax(aBlock[i_local][j_local], aBlock[i_local][j_local + 16]),
            fmax(aBlock[i_local + 16][j_local], aBlock[i_local + 16][j_local + 16]));
        __syncthreads();
    }
    if (blockDim.x >= 16 && blockDim.y >= 16 && i_local < 8 && j_local < 8) {
        aBlock[i_local][j_local] = fmax(fmax(aBlock[i_local][j_local], aBlock[i_local][j_local + 8]),
            fmax(aBlock[i_local + 8][j_local], aBlock[i_local + 8][j_local + 8]));
        __syncthreads();
    }
    if (blockDim.x >= 8 && blockDim.y >= 8 && i_local < 4 && j_local < 4) {
        aBlock[i_local][j_local] = fmax(fmax(aBlock[i_local][j_local], aBlock[i_local][j_local + 4]),
            fmax(aBlock[i_local + 4][j_local], aBlock[i_local + 4][j_local + 4]));
        __syncthreads();
    }
    if (blockDim.x >= 4 && blockDim.y >= 4 && i_local < 2 && j_local < 2) {
        aBlock[i_local][j_local] = fmax(fmax(aBlock[i_local][j_local], aBlock[i_local][j_local + 2]),
            fmax(aBlock[i_local + 2][j_local], aBlock[i_local + 2][j_local + 2]));
        __syncthreads();
    }
    if (blockDim.x >= 2 && blockDim.y >= 2 && i_local < 1 && j_local < 1) {
        aBlock[i_local][j_local] = fmax(fmax(aBlock[i_local][j_local], aBlock[i_local][j_local + 1]),
            fmax(aBlock[i_local + 1][j_local], aBlock[i_local + 1][j_local + 1]));
        __syncthreads();
    }

    // a_max in current block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        aDevice[blockIdx.y * gridDim.x + blockIdx.x] = aBlock[threadIdx.x][threadIdx.y];
    }
}


// function: calculate the velocity in each cell on GPU
__global__ void computeAmax(double* aDevice, Grid u) {

    // variable substitution
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double cur_a = 0.0;

    // calculate a
    if (i >= nGhost && i < u.nCellsX + nGhost && j >= nGhost && j < u.nCellsY + nGhost) {
        double u_prim[NUM_VARS];
        double u_cons[NUM_VARS];
        for (int v = 0; v < NUM_VARS; v++) {
            u_cons[v] = u(i, j, v);
        }
        cons2primDevice(u_prim, u_cons);
        double cur_rho = u_prim[0], cur_vx = u_prim[1], cur_vy = u_prim[2], cur_p = u_prim[3];
        double cur_v = pow(pow(cur_vx, 2) + pow(cur_vy, 2), 0.5);
        double Cs = pow(gamma * cur_p / cur_rho, 0.5);
        cur_a = cur_v + Cs;
    }

    aDevice[j * blockDim.x * gridDim.x + i] = cur_a;
}


// function: calculate time step
double computeTimeStep(Grid u, const double& dx, const double& dy, const dim3& dimGrid, const dim3& dimBlock, bool optTime) {

    // calculate a's on GPU
    double* aDevice;
    int a_size = 0;
    if (optTime) {  // with shared memory optimization
        a_size = dimGrid.x * dimGrid.y;
        cudaMalloc(&aDevice, a_size * sizeof(double));
        computeAmaxOpt<<<dimGrid, dimBlock>>>(aDevice, u);
        CUDA_CHECK;
    } else {  // without shared memory optimization
        a_size = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y;
        cudaMalloc(&aDevice, a_size * sizeof(double));
        computeAmax<<<dimGrid, dimBlock>>>(aDevice, u);
        CUDA_CHECK;
    }

    // transfer data to CPU
    double* aHost = new double [a_size];
    cudaMemcpy(aHost, aDevice, a_size * sizeof(double), cudaMemcpyDeviceToHost);

    // for stability: numerical dependence stencil should contain the largest wave speed
    double a_max = 0.0;
    for (int i = 0; i < a_size; i++) {
        a_max = std::max(a_max, aHost[i]);
    }
    cudaFree(aDevice);
    delete[] aHost;
    const double timeStep = C * std::min(dx, dy) / a_max;
    return timeStep;
}


// function: calculate slope limiter
__device__ double getLimiter(double r) {

    // // Minbee
    // if (r <= 0) {return 0.0;}
    // if (r > 0 && r <= 1) {return r;}
    // if (r > 1) {return fmin(1.0, 2.0 / (1 + r));}

    // Superbee
    if (r <= 0.0) {return 0.0;}
    if (r > 0.0 and r <= 0.5) {double res = 2 * r; return res;}
    if (r > 0.5 and r <= 1.0) {return 1.0;}
    if (r > 1.0) {double temp = fmin(r, 2.0 / (1 + r)); return fmin(temp, 2.0);}

    return 0;
}


// function: data reconstruction for a single cell
__device__ void dataReconstruct(double* u_backward, double* u_forward, double* u) {

    for (int v = 0; v < NUM_VARS; v++) {
        double q0 = u_backward[v], q = u[v], q1 = u_forward[v];

        double r = (q - q0) / (q1 - q);
        double slope_limiter = getLimiter(r);
        // double slope_limiter = 0.0;

        double delta_backward = q - q0;
        double delta_forward = q1 - q;
        double delta_i = 0.5 * (delta_backward + delta_forward);

        double qBarBackward = q - 0.5 * slope_limiter * delta_i;
        double qBarForward = q + 0.5 * slope_limiter * delta_i;

        u_backward[v] = qBarBackward;
        u_forward[v] = qBarForward;
    }
}


// function: calculate flux functions
template<int axis>
__device__ void flux_func(double* flux, const double* u_cons) {

    double u_prim[NUM_VARS];
    cons2primDevice(u_prim, u_cons);
    double rho = u_cons[0], momx = u_cons[1], momy = u_cons[2], E = u_cons[3];
    double vx = u_prim[1], vy = u_prim[2], p = u_prim[3];

    double rho_flux = axis == 0 ? momx : momy;
    double momx_flux = axis == 0 ? rho * pow(vx, 2) + p : rho * vx * vy;
    double momy_flux = axis == 0 ? rho * vy * vx : rho * pow(vy, 2) + p;
    double E_flux = axis == 0 ? (E + p) * vx : (E + p) * vy;

    flux[0] = rho_flux;
    flux[1] = momx_flux;
    flux[2] = momy_flux;
    flux[3] = E_flux;
}


// function: half-time step update
template<int axis>
__device__ void halfTimeStepUpdate(double* u_backward, double* u_forward, const double dx, const double dy, const double dt) {

    // calculate flux functions
    double flux_f[NUM_VARS], flux_b[NUM_VARS];
    flux_func<axis>(flux_b, u_backward);
    flux_func<axis>(flux_f, u_forward);

    // update
    double unit_len = axis == 0 ? dx : dy;
    for (int v = 0; v < NUM_VARS; v++) {
        double flux_update = 0.5 * (dt / unit_len) * (flux_f[v] - flux_b[v]);
        u_backward[v] = u_backward[v] - flux_update;
        u_forward[v] = u_forward[v] - flux_update;
    }
}


// function: calculate numerical fluxes with FORCE scheme
template<int axis>
__device__ void calFlux(double* flux, double* u_backward, double* u_forward, const double dx, const double dy, const double dt) {

    // calculate flux functions
    double flux_f[NUM_VARS], flux_b[NUM_VARS];
    flux_func<axis>(flux_b, u_backward);
    flux_func<axis>(flux_f, u_forward);

    // L-F scheme
    double unit_len = axis == 0 ? dx : dy;
    const double F_rho_LF = 0.5 * unit_len / dt * (u_backward[0] - u_forward[0]) + 0.5 * (flux_b[0] + flux_f[0]);
    const double F_momx_LF = 0.5 * unit_len / dt * (u_backward[1] - u_forward[1]) + 0.5 * (flux_b[1] + flux_f[1]);
    const double F_momy_LF = 0.5 * unit_len / dt * (u_backward[2] - u_forward[2]) + 0.5 * (flux_b[2] + flux_f[2]);
    const double F_E_LF = 0.5 * unit_len / dt * (u_backward[3] - u_forward[3]) + 0.5 * (flux_b[3] + flux_f[3]);

    // RI scheme
    double u_half_cons[NUM_VARS]; double F_RI[NUM_VARS];
    u_half_cons[0] = 0.5 * (u_backward[0] + u_forward[0]) - 0.5 * dt / unit_len * (flux_f[0] - flux_b[0]);
    u_half_cons[1] = 0.5 * (u_backward[1] + u_forward[1]) - 0.5 * dt / unit_len * (flux_f[1] - flux_b[1]);
    u_half_cons[2] = 0.5 * (u_backward[2] + u_forward[2]) - 0.5 * dt / unit_len * (flux_f[2] - flux_b[2]);
    u_half_cons[3] = 0.5 * (u_backward[3] + u_forward[3]) - 0.5 * dt / unit_len * (flux_f[3] - flux_b[3]);
    flux_func<axis>(F_RI, u_half_cons);

    // FORCE scheme
    const double F_rho_FORCE = 0.5 * (F_rho_LF + F_RI[0]);
    const double F_momx_FORCE = 0.5 * (F_momx_LF + F_RI[1]);
    const double F_momy_FORCE = 0.5 * (F_momy_LF + F_RI[2]);
    const double F_E_FORCE = 0.5 * (F_E_LF + F_RI[3]);
    flux[0] = F_rho_FORCE;
    flux[1] = F_momx_FORCE;
    flux[2] = F_momy_FORCE;
    flux[3] = F_E_FORCE;
}


// function: SLIC evolution in a single kernel with shared memory optimization
__global__ void SLIC_Evolution_X(Grid u, const double dx, const double dy, const double dt) {

    // allocate shared memory
    __shared__ double uL[nThreadsXSLICX][nThreadsYSLICX][NUM_VARS];
    __shared__ double uI[nThreadsXSLICX][nThreadsYSLICX][NUM_VARS];
    __shared__ double uR[nThreadsXSLICX][nThreadsYSLICX][NUM_VARS];
    __shared__ double flux[nThreadsXSLICX][nThreadsYSLICX][NUM_VARS];

    // get coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x - 2 * blockIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_local = threadIdx.x, j_local = threadIdx.y;
    if (i_local >= nThreadsXSLICX || j_local >= nThreadsYSLICX) {assert(false);}

    // data reconstruction and half-time update
    int i_min = nGhost - 1;
    int i_max = u.nCellsX + nGhost + 1;
    int j_min = nGhost;
    int j_max = u.nCellsY + nGhost;
    if (i >= i_min && i < i_max && j >= j_min && j < j_max) {

        // read data from global memory
        for (int v = 0; v < NUM_VARS; v++) {
            uI[i_local][j_local][v] = u(i, j, v);
        }
        __syncthreads();

        // read uL and uR
        for (int v = 0; v < NUM_VARS; v++) {
            if (i_local == 0) {
                uL[i_local][j_local][v] = u(i - 1, j, v);
            } else {
                uL[i_local][j_local][v] = uI[i_local - 1][j_local][v];
            }
            if (i_local == blockDim.x - 1) {
                uR[i_local][j_local][v] = u(i + 1, j, v);
            } else {
                uR[i_local][j_local][v] = uI[i_local + 1][j_local][v];
            }
        }

        // data reconstruction
        dataReconstruct(uL[i_local][j_local], uR[i_local][j_local], uI[i_local][j_local]);
        // half time-step update
        halfTimeStepUpdate<0>(uL[i_local][j_local], uR[i_local][j_local], dx, dy, dt);
        __syncthreads();
    }

    // calculate fluxes
    i_min = nGhost - 1;
    i_max = u.nCellsX + nGhost;
    j_min = nGhost;
    j_max = u.nCellsY + nGhost;
    if (i >= i_min && i < i_max && j >= j_min && j < j_max) {
        if (i_local < blockDim.x - 1) {
            calFlux<0>(flux[i_local][j_local], uR[i_local][j_local], uL[i_local + 1][j_local], dx, dy, dt);
        }
        __syncthreads();
    }

    // evolution by adding fluxes
    i_min = nGhost;
    i_max = u.nCellsX + nGhost;
    j_min = nGhost;
    j_max = u.nCellsY + nGhost;
    if (i >= i_min && i < i_max && j >= j_min && j < j_max) {
        if (i_local > 0 && i_local < blockDim.x - 1) {
            for (int v = 0; v < NUM_VARS; v++) {
                u(i, j, v) = u(i, j, v) - dt / dx * (flux[i_local][j_local][v] - flux[i_local - 1][j_local][v]);
            }
        }
        __syncthreads();
    }
}


__global__ void SLIC_Evolution_Y(Grid u, const double dx, const double dy, const double dt) {

    // allocate shared memory
    __shared__ double uL[nThreadsXSLICY][nThreadsYSLICY][NUM_VARS];
    __shared__ double uI[nThreadsXSLICY][nThreadsYSLICY][NUM_VARS];
    __shared__ double uR[nThreadsXSLICY][nThreadsYSLICY][NUM_VARS];
    __shared__ double flux[nThreadsXSLICY][nThreadsYSLICY][NUM_VARS];

    // get coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y - 2 * blockIdx.y;
    int i_local = threadIdx.x, j_local = threadIdx.y;

    // data reconstruction and half-time update
    int i_min = nGhost;
    int i_max = u.nCellsX + nGhost;
    int j_min = nGhost - 1;
    int j_max = u.nCellsY + nGhost + 1;
    if (i >= i_min && i < i_max && j >= j_min && j < j_max) {

        // read data from global memory
        for (int v = 0; v < NUM_VARS; v++) {
            uI[i_local][j_local][v] = u(i, j, v);
        }
        __syncthreads();

        // read uL and uR
        for (int v = 0; v < NUM_VARS; v++) {
            if (j_local == 0) {
                uL[i_local][j_local][v] = u(i, j - 1, v);
            } else {
                uL[i_local][j_local][v] = uI[i_local][j_local - 1][v];
            }
            if (j_local == blockDim.y - 1) {
                uR[i_local][j_local][v] = u(i, j + 1, v);
            } else {
                uR[i_local][j_local][v] = uI[i_local][j_local + 1][v];
            }
        }

        // data reconstruction
        dataReconstruct(uL[i_local][j_local], uR[i_local][j_local], uI[i_local][j_local]);
        // half time-step update
        halfTimeStepUpdate<1>(uL[i_local][j_local], uR[i_local][j_local], dx, dy, dt);
        __syncthreads();
    }

    // calculate fluxes
    i_min = nGhost;
    i_max = u.nCellsX + nGhost;
    j_min = nGhost - 1;
    j_max = u.nCellsY + nGhost;
    if (i >= i_min && i < i_max && j >= j_min && j < j_max) {
        if (j_local < blockDim.y - 1) {
            calFlux<1>(flux[i_local][j_local], uR[i_local][j_local], uL[i_local][j_local + 1], dx, dy, dt);
        }
        __syncthreads();
    }

    // evolution by adding fluxes
    i_min = nGhost;
    i_max = u.nCellsX + nGhost;
    j_min = nGhost;
    j_max = u.nCellsY + nGhost;
    if (i >= i_min && i < i_max && j >= j_min && j < j_max) {
        if (j_local > 0 && j_local < blockDim.y - 1) {
            for (int v = 0; v < NUM_VARS; v++) {
                u(i, j, v) = u(i, j, v) - dt / dy * (flux[i_local][j_local][v] - flux[i_local][j_local - 1][v]);
            }
        }
        __syncthreads();
    }
}


// function: data recording
void dataRecord(Grid uHost, Grid u, const int case_id, const double nCellsX, const double nCellsY,
    const double x0, const double y0, const double dx, const double dy, const double t) {

    // copy data from GPU to CPU
    cudaMemcpy(uHost.data, u.data, (nCellsX + 2 * nGhost) * (nCellsY + 2 * nGhost) * NUM_VARS * sizeof(double),
        cudaMemcpyDeviceToHost);

    // check whether the directory exists, create one if not
    std::ostringstream folderPath;
    folderPath << "res/Case_" << case_id;
    std::string caseFolder = folderPath.str();
    if (!fs::exists(caseFolder)) {
        fs::create_directories(caseFolder);
    }

    // data recording
    std::ostringstream oss;
    oss << caseFolder << "/T=" << std::setprecision(2) << t << ".txt";
    std::string fileName = oss.str();
    std::fstream outFile(fileName, std::ios::out);

    double* u_ij_prim = new double[NUM_VARS];
    double* u_ij_cons = new double[NUM_VARS];

    for (int i = nGhost; i < nCellsX + nGhost; i++) {
        for (int j = nGhost; j < nCellsY + nGhost; j++) {
            for (int v = 0; v < NUM_VARS; v++) {
                u_ij_cons[v] = uHost(i, j, v);
            }
            cons2primHost(u_ij_prim, u_ij_cons);

            outFile << x0 + (i - nGhost + 0.5) * dx << ", " << y0 + (j - nGhost + 0.5) * dy
            << ", " << u_ij_prim[0] << ", " << u_ij_prim[1] << ", " << u_ij_prim[2] << ", " << u_ij_prim[3]
            << std::endl;
        }
    }
    outFile.close();
}


// function: initialization
__global__ void initialize(Grid u, const double x0, const double y0, const double dx, const double dy,
    const int case_id, const double bubble_center_x, double bubble_center_y, double bubble_radius) {

    // get coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nGhost && i < u.nCellsX + nGhost && j >= nGhost && j < u.nCellsY + nGhost) {
        double x = x0 + (i - nGhost + 0.5) * dx;
        double y = y0 + (j - nGhost + 0.5) * dy;
        double u_ij_prim[NUM_VARS];
        double u_ij_cons[NUM_VARS];

        if (case_id == 1) {
            if (x >= 0.5 && y >= 0.5) {
                u_ij_prim[0] = 1.5;
                u_ij_prim[1] = 0.0;
                u_ij_prim[2] = 0.0;
                u_ij_prim[3] = 1.5;
            }
            if (x < 0.5 && y >= 0.5) {
                u_ij_prim[0] = 0.5325;
                u_ij_prim[1] = 1.206;
                u_ij_prim[2] = 0.0;
                u_ij_prim[3] = 0.3;
            }
            if (x < 0.5 && y < 0.5) {
                u_ij_prim[0] = 0.138;
                u_ij_prim[1] = 1.206;
                u_ij_prim[2] = 1.206;
                u_ij_prim[3] = 0.029;
            }
            if (x >= 0.5 && y < 0.5) {
                u_ij_prim[0] = 0.5325;
                u_ij_prim[1] = 0.0;
                u_ij_prim[2] = 1.206;
                u_ij_prim[3] = 0.3;
            }
        }

        if (case_id == 2) {
            if (x < 0.005) {
                u_ij_prim[0] = 1.7755;
                u_ij_prim[1] = 110.63;
                u_ij_prim[2] = 0.0;
                u_ij_prim[3] = 159060.0;
            }
            else if (pow(pow(x - bubble_center_x, 2) + pow(y - bubble_center_y, 2), 0.5) <= bubble_radius) {
                u_ij_prim[0] = 0.214;
                u_ij_prim[1] = 0.0;
                u_ij_prim[2] = 0.0;
                u_ij_prim[3] = 101325.0;
            }
            else {
                u_ij_prim[0] = 1.29;
                u_ij_prim[1] = 0.0;
                u_ij_prim[2] = 0.0;
                u_ij_prim[3] = 101325.0;
            }
        }

        // transform from primitive to conservative and store
        prim2consDevice(u_ij_cons, u_ij_prim);
        u(i, j, 0) = u_ij_cons[0];
        u(i, j, 1) = u_ij_cons[1];
        u(i, j, 2) = u_ij_cons[2];
        u(i, j, 3) = u_ij_cons[3];
    }
}


// function: check memory usage
void checkKernelAttributes() {
    cudaFuncAttributes attr;

    cudaFuncGetAttributes(&attr, computeAmaxOpt);
    std::cout << "=== Kernel Resource Usage: computeAmaxOpt ===" << std::endl;
    std::cout << "Registers used: " << attr.numRegs << std::endl;
    std::cout << "Shared memory per block: " << attr.sharedSizeBytes << " bytes" << std::endl;
    std::cout << "Constant memory used: " << attr.constSizeBytes << " bytes" << std::endl;
    std::cout << "Local memory per thread: " << attr.localSizeBytes << " bytes" << std::endl;
    std::cout << "Max threads per block: " << attr.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;

    cudaFuncGetAttributes(&attr, SLIC_Evolution_X);
    std::cout << "=== Kernel Resource Usage: SLIC_Evolution_X ===" << std::endl;
    std::cout << "Registers used: " << attr.numRegs << std::endl;
    std::cout << "Shared memory per block: " << attr.sharedSizeBytes << " bytes" << std::endl;
    std::cout << "Constant memory used: " << attr.constSizeBytes << " bytes" << std::endl;
    std::cout << "Local memory per thread: " << attr.localSizeBytes << " bytes" << std::endl;
    std::cout << "Max threads per block: " << attr.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;

    cudaFuncGetAttributes(&attr, SLIC_Evolution_Y);
    std::cout << "=== Kernel Resource Usage: SLIC_Evolution_Y ===" << std::endl;
    std::cout << "Registers used: " << attr.numRegs << std::endl;
    std::cout << "Shared memory per block: " << attr.sharedSizeBytes << " bytes" << std::endl;
    std::cout << "Constant memory used: " << attr.constSizeBytes << " bytes" << std::endl;
    std::cout << "Local memory per thread: " << attr.localSizeBytes << " bytes" << std::endl;
    std::cout << "Max threads per block: " << attr.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;
}


// function: mainloop
int main() {

    // experimental options
    int case_id = 2;  // Case 1: Quadrant problem; Case 2: Shock-Bubble interaction
    bool Record = false;  // whether record experimental data
    bool optTime = true;  // whether optimize dt calculation with shared memory
    double bubble_center_x = 0.035, bubble_center_y = 0.0445, bubble_radius = 0.025;
    double Ms = 1.22, p_Air = 101325.0, rho_Air = 1.29;
    double Cs_Air = pow(gamma * p_Air / rho_Air, 0.5);
    double time_ratio = bubble_radius / (Cs_Air * Ms);
    if (case_id == 1) {time_ratio = 1;}

    // parameters
    std::array<int, 2> nCellsX_list = {400, 500};
    std::array<int, 2> nCellsY_list = {400, 197};
    std::array<double, 2> x1_list = {1.0, 0.225};
    std::array<double, 2> y1_list = {1.0, 0.089};
    std::array<double, 2> tStop_list = {0.3, 7.8 * time_ratio};

    double x0 = 0.0, y0 = 0.0, tStart = 0.0;
    int nCellsX = nCellsX_list[case_id - 1], nCellsY = nCellsY_list[case_id - 1];
    double x1 = x1_list[case_id - 1], y1 = y1_list[case_id - 1];
    double dx = (x1 - x0) / nCellsX, dy = (y1 - y0) / nCellsY;
    double tStop = tStop_list[case_id - 1];

    int nBlocksX = (nCellsX + 2 * nGhost + nThreadsX - 1) / nThreadsX;
    int nBlocksY = (nCellsY + 2 * nGhost + nThreadsY - 1) / nThreadsY;
    dim3 dimBlock(nThreadsX, nThreadsY, 1);
    dim3 dimGrid(nBlocksX, nBlocksY, 1);

    // x-direction evolution with overlapping blocks
    int nBlocksXSLICX = (nCellsX + 2 * nGhost - nThreadsXSLICX + nThreadsXSLICX - 3) / (nThreadsXSLICX - 2) + 1;
    int nBlocksYSLICX = (nCellsY + 2 * nGhost + nThreadsYSLICX - 1) / nThreadsYSLICX;
    dim3 dimBlockSLICX(nThreadsXSLICX, nThreadsYSLICX, 1);
    dim3 dimGridSLICX(nBlocksXSLICX, nBlocksYSLICX, 1);

    // y-direction evolution with overlapping blocks
    int nBlocksXSLICY = (nCellsX + 2 * nGhost + nThreadsXSLICY - 1) / nThreadsXSLICY;
    int nBlocksYSLICY = (nCellsY + 2 * nGhost - nThreadsYSLICY + nThreadsYSLICY - 3) / (nThreadsYSLICY - 2) + 1;
    dim3 dimBlockSLICY(nThreadsXSLICY, nThreadsYSLICY, 1);
    dim3 dimGridSLICY(nBlocksXSLICY, nBlocksYSLICY, 1);

    // execution time recording
    double elapsdt = 0, elapsx = 0, elapsy = 0, elapsbc = 0, elapstotal = 0;
    clock_t startx, endx, starty, endy, startdt, enddt, startbc, endbc, start, end;

    // initialization
    Grid uHost(nCellsX, nCellsY, CPU);  // data on CPU for recording
    Grid u(nCellsX, nCellsY, GPU);  // data in conservative form on GPU
    initialize<<<dimGrid, dimBlock>>>(u, x0, y0, dx, dy, case_id, bubble_center_x, bubble_center_y, bubble_radius);
    CUDA_CHECK;

    // boundary conditions
    setBoundaryCondition<<<dimGrid, dimBlock>>>(u);
    CUDA_CHECK;

    // update data
    double t = tStart;
    int counter = 0;
    do {
        start = clock();
        // check memory usage
        if (counter == 0) {checkKernelAttributes();}

        // compute time step
        startdt = clock();
        double dt = computeTimeStep(u, dx, dy, dimGrid, dimBlock, optTime);
        enddt = clock();
        elapsdt += (double)(enddt - startdt) / CLOCKS_PER_SEC;

        t = t + dt;
        counter++;
        std::cout << "ite = " << counter<< ", time = " << t << std::endl;

        // x-direction evolution
        startx = clock();
        SLIC_Evolution_X<<<dimGridSLICX, dimBlockSLICX>>>(u, dx, dy, dt);
        CUDA_CHECK;
        endx = clock();
        elapsx += (double)(endx - startx) / CLOCKS_PER_SEC;

        // boundary conditions
        startbc = clock();
        setBoundaryCondition<<<dimGrid, dimBlock>>>(u);
        CUDA_CHECK;
        endbc = clock();
        elapsbc += (double)(endbc - startbc) / CLOCKS_PER_SEC;

        // y-direction evolution
        starty = clock();
        SLIC_Evolution_Y<<<dimGridSLICY, dimBlockSLICY>>>(u, dx, dy, dt);
        CUDA_CHECK;
        endy = clock();
        elapsy += (double)(endy - starty) / CLOCKS_PER_SEC;

        // boundary conditions
        startbc = clock();
        setBoundaryCondition<<<dimGrid, dimBlock>>>(u);
        CUDA_CHECK;
        endbc = clock();
        elapsbc += (double)(endbc - startbc) / CLOCKS_PER_SEC;

        end = clock();
        elapstotal += (double)(end - start) / CLOCKS_PER_SEC;

    } while (t < tStop);

    // data recording
    if (Record) {
        std::cout << "Recording: t = " << t << std::endl;
        dataRecord(uHost, u, case_id, nCellsX, nCellsY, x0, y0, dx, dy, t / time_ratio);
    }

    // release memory
    cudaFree(u.data);
    delete[] uHost.data;

    // output time recording
    std::cout << "=== Timing Results ===" << std::endl;
    std::cout << "Total execution time: " << elapstotal << " sec" << std::endl;
    std::cout << "computeTimeStep: " << elapsdt << " sec" << std::endl;
    std::cout << "Boundary Conditions: " << elapsbc << " sec" << std::endl;
    std::cout << "X-direction evolution: " << elapsx << " sec" << std::endl;
    std::cout << "Y-direction evolution: " << elapsy << " sec" << std::endl;

    return 0;
}

