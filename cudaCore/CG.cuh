#include <cuda_runtime.h>
#include "device_launch_parameters.h"
// #include "helper_math.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <stdlib.h>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/sort.h>
#include<thrust/execution_policy.h>

#include<cooperative_groups.h>
#include<cooperative_groups/reduce.h>

#include "../Math/Solver.hpp"

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512

#ifndef WITH_GRAPH
#define WITH_GRAPH 1
#endif

inline void processCudaError(cudaError_t err, const char *file, int line) {
    if (err == cudaSuccess)
        return;
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " happened at line: " << line
              << " in file: " << file << std::endl;
}
#define CUDACheck(f) processCudaError(f, __FILE__, __LINE__)

inline void processCUSPARSEError(cusparseStatus_t err, const char *file, int line) {
    if (err == CUSPARSE_STATUS_SUCCESS)
        return;
    std::cerr << "CUSPARSE error: " << cusparseGetErrorString(err) << " happened at line: " << line
              << " in file: " << file << std::endl;
}
#define CUSPARSECheck(f) processCUSPARSEError(f, __FILE__, __LINE__)

inline void processCUBLASError(cublasStatus_t err, const char *file, int line) {
    if (err == CUBLAS_STATUS_SUCCESS)
        return;
    std::cerr << "CUBLAS error: " << (int)err << " happened at line: " << line
              << " in file: " << file << std::endl;
}
#define CUBLASCheck(f) processCUBLASError(f, __FILE__, __LINE__)


typedef struct CudaVector
{   
    cusparseDnVecDescr_t vec;
    Scalar *ptr;
} Vec;

struct MultiDeviceData {
    unsigned char *hostMemoryArrivedList;
    unsigned int numDevices;
    unsigned int deviceRank;
};

class PeerGroup {
private:
    const MultiDeviceData &data;
    const cg::grid_group &grid;

    __device__ unsigned char load_arrived(unsigned char *arrived) const {
#if __CUDA_ARCH__ < 700
        return *(volatile unsigned char *)arrived;
#else   
        unsigned int result;
        asm volatile("ld.acquire.sys.global.u8 %0, [%1];"
                    : "=r"(result)
                    : "l"(arrived)
                    : "memory");
        return result;
#endif
    }

    __device__ void store_arrived(unsigned char *arrived, unsigned char val) const {
#if __CUDA_ARCH__ < 700
        *(volatile unsigned char *)arrived = val;
#else
        unsigned int reg_val = val;
        asm volatile("st.release.sys.global.u8 [%0], %1;"
                    :
                    : "r"(reg_val) "l"(arrived) 
                    : "memory");
        // Avoids compiler warnings from unused variable val
        (void)(reg_val = reg_val);
#endif
    }

public:
    __device__ PeerGroup(const MultiDeviceData &data, const cg::grid_group &grid) : data(data), grid(grid) {}

    __device__ unsigned int size() const {
        return data.numDevices * grid.size();
    }

    __device__ unsigned int thread_rank() const {
        return data.deviceRank * grid.size() + grid.thread_rank();
    }

    __device__ void sync() const {
        grid.sync();

        // One thread from each grid participates in the sync
        if (grid.thread_rank() == 0) {
            if (data.deviceRank == 0) {
                // Leader grid waits for others to join and then releases them.
                // Other GPUs can arrive in any order, so the leader have to wait for all others.
                for (int i = 0; i < data.numDevices - 1; i++) {
                    while (load_arrived(&data.hostMemoryArrivedList[i]) == 0) 
                        ;
                }
                for (int i = 0; i < data.numDevices - 1; i++) {
                    store_arrived(&data.hostMemoryArrivedList[i], 0);
                }
                __threadfence_system();
            } else {
                // Other grids note their arrival and wait to be released.
                store_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1], 1);
                while (load_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1]) == 1) 
                    ;
            }
        }

        grid.sync();
    }
};

void CG(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void CG_CG(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void PCG_ICC(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void BiCGSTAB(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void CG_UM(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void CG_MBCG(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void CG_MGCG(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);