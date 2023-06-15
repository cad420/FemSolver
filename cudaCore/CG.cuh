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

#include "../Math/Solver.hpp"

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

void CG(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void PCG_ICC(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);

void BiCGSTAB(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm);
