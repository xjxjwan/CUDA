//
// Created by Lenovo on 25-03-01.
//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <array>

// macro: constant parameters
#define NUM_VARS 4  // number of independent variables
#define nGhost 2  // number of ghost cells in each direction
#define C 0.8  // CFL number
#define gamma 1.4  // adiabatic index of ideal gas
#define nThreadsX 32  // number of threads per block in x-direction
#define nThreadsY 32  // number of threads per block in y-direction
#define nThreadsXOverlap 6  // number of threads per block in x-direction in SLIC
#define nThreadsYOverlap 6  // number of threads per block in y-direction in SLIC

// macro: debugging
#define CUDA_CHECK {\
    cudaDeviceSynchronize();\
    cudaError_t err = cudaGetLastError();\
    if(err){\
        std::cout << "Error: " << cudaGetErrorString(err) << " line " << __LINE__ << std::endl;\
        exit(1);\
    }\
}


// structure: data storage and process
struct Grid {
    double *data;
    int nCellsX, nCellsY;
    double x0, x1, y0, y1;

    // format for building an instance
    Grid(const int nX, const int nY, const std::array<double, 4>& sim_range) {
        nCellsX = nX;
        nCellsY = nY;
        x0 = sim_range[0], x1 = sim_range[1];
        y0 = sim_range[2], y1 = sim_range[3];
        cudaMalloc((void **)& data, (nCellsX + 2 * nGhost) * (nCellsY + 2 * nGhost) * NUM_VARS * sizeof(double));
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
std::array<double, 4> prim2consHost(std::array<double, 4> const& u_ij) {

    const double rho = u_ij[0];
    const double u = u_ij[1];
    const double v = u_ij[2];
    const double p = u_ij[3];

    std::array<double, 4> res{};
    res[0] = rho;  // rho
    res[1] = rho * u;  // momx
    res[2] = rho * v;  // momy
    res[3] = p / (gamma - 1) + 0.5 * rho * (pow(u, 2) + pow(v, 2));  // E

    return res;
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
std::array<double, 4> cons2primHost(std::array<double, 4> const& u_ij) {

    const double rho = u_ij[0];
    const double momx = u_ij[1];
    const double momy = u_ij[2];
    const double E = u_ij[3];

    std::array<double, 4> res{};
    res[0] = rho;  // rho
    res[1] = momx / rho;  // u
    res[2] = momy / rho;  // v
    res[3] = (gamma - 1) * (E - 0.5 * pow(momx, 2) / rho - 0.5 * pow(momy, 2) / rho);  // p

    return res;
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


// function: calculate the maximum velocity in the whole grid on GPU
__global__ void computeAmax(double* aDevice, Grid u) {

    // shared memory for current block
    __shared__ double aBlock[nThreadsX][nThreadsY];

    // variable substitution
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_local = threadIdx.x, j_local = threadIdx.y;
    double cur_v = 0.0;

    // calculate a's in current block
    if (i >= nGhost && i < u.nCellsX + nGhost && j >= nGhost && j < u.nCellsY + nGhost) {
        double u_prim[NUM_VARS];
        double u_cons[NUM_VARS];
        for (int v = 0; v < NUM_VARS; v++) {
            u_cons[v] = u(i, j, v);
        }
        cons2primDevice(u_prim, u_cons);
        double cur_vx = u_prim[1], cur_vy = u_prim[2];
        cur_v = pow(pow(cur_vx, 2) + pow(cur_vy, 2), 0.5);
    }
    aBlock[threadIdx.x][threadIdx.y] = cur_v;
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


// function: calculate time step
double computeTimeStep(const Grid& u, const double& dx, const double& dy, const dim3& dimGrid, const dim3& dimBlock) {

    // calculate a_max on GPU
    double* aDevice;
    int a_size = dimGrid.x * dimGrid.y;
    cudaMalloc(&aDevice, a_size * sizeof(double));
    computeAmax<<<dimGrid, dimBlock>>>(aDevice, u);
    CUDA_CHECK;

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


// function: initialization
__global__ void initialize(Grid u, const double x0, const double y0, const double y1,
    const double dx, const double dy, const int case_id) {

    // get coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double bubble_center_x = 35;
    double bubble_center_y = 0.5 * y1;
    double bubble_radius = 25;

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
            if (x < 5) {
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


// function: mainloop
int main() {

    // parameters
    int case_id = 1;
    std::array<int, 2> nCellsX_list = {400, 500};
    std::array<int, 2> nCellsY_list = {400, 197};
    std::array<double, 2> x1_list = {1.0, 225.0};
    std::array<double, 2> y1_list = {1.0, 89.0};
    std::array<double, 2> tStop_list = {0.3, 0.3};

    double x0 = 0.0, y0 = 0.0, tStart = 0.0;
    int nCellsX = nCellsX_list[case_id - 1], nCellsY = nCellsY_list[case_id - 1];
    double x1 = x1_list[case_id - 1], y1 = y1_list[case_id - 1];
    double dx = (x1 - x0) / nCellsX, dy = (y1 - y0) / nCellsY;
    double tStop = tStop_list[case_id - 1];

    // initialization
    std::array<double, 4> sim_range = {x0, x1, y0, y1};
    Grid u(nCellsX, nCellsY, sim_range);  // in conservative form

    int nBlocksX = (int)ceil((nCellsX + 2 * nGhost) / nThreadsX);
    int nBlocksY = (int)ceil((nCellsY + 2 * nGhost) / nThreadsY);
    dim3 dimBlock(nThreadsX, nThreadsY, 1);
    dim3 dimGrid(nBlocksX, nBlocksY, 1);

    initialize<<<dimGrid, dimBlock>>>(u, x0, y0, y1, dx, dy, case_id);
    CUDA_CHECK;

    // boundary conditions
    setBoundaryCondition<<<dimGrid, dimBlock>>>(u);
    CUDA_CHECK;

    // update data
    double t = tStart;
    // std::array t_record_list = {0.1, 0.2, 0.3};
    // int record_index = 0;
    int counter = 0;
    do {
        // compute time step
        double dt = computeTimeStep(u, dx, dy, dimGrid, dimBlock);
        t = t + dt;
        std::cout << "ite = " << counter + 1 << ", time = " << t << std::endl;

    } while (t < tStop);

    return 0;
}

