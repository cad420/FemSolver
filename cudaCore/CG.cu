#include "CG.cuh"

void CG(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
{
    printf("CG...\n");
    auto m_Mat = A.getMat();
	int num_rows = m_Mat.size(), nnz = 0;
	int m = num_rows;
	int num_offsets = m + 1;

    int*    h_A_rows    = (int*)    malloc(num_offsets * sizeof(int));
    h_A_rows[0] = 0;
	for (int i = 0; i < m_Mat.size(); i++) {
		for (int j = 0; j < m_Mat[i].size(); j++) {
			if (m_Mat[i][j].second != 0) {
				nnz++;
			}
		}
        h_A_rows[i + 1] = nnz;
	}
	
    int*    h_A_columns = (int*)    malloc(nnz * sizeof(int));
    double* h_A_values  = (double*) malloc(nnz * sizeof(double));
    double* h_X         = (double*) malloc(m * sizeof(double));
	// from ellpack to csr
    int cnt = 0;
    for (int i = 0; i < m_Mat.size(); i++) {
		for (int j = 0; j < m_Mat[i].size(); j++) {
			if (m_Mat[i][j].second != 0) {
				h_A_columns[cnt] = m_Mat[i][j].first;
                h_A_values[cnt] = m_Mat[i][j].second;
                cnt++;
			}
		}
	}
	
    for (int i = 0; i < num_rows; i++)
        h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int*    d_A_rows, *d_A_columns;
    double* d_A_values;
    Vec     d_B, d_X, d_R, d_P, d_T;

    // allocate device memory for CSR matrices
    CUDACheck( cudaMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) );
    CUDACheck( cudaMalloc((void**) &d_A_columns, nnz * sizeof(int)) );
    CUDACheck( cudaMalloc((void**) &d_A_values,  nnz * sizeof(double)) );
    
    CUDACheck( cudaMalloc((void**) &d_B.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_X.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_R.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_P.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_T.ptr,     m * sizeof(double)) );
    
    // copy the CSR matrices and vectors into device memory
    CUDACheck( cudaMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_X.ptr, h_X, m * sizeof(double),
                           cudaMemcpyHostToDevice) );
    //--------------------------------------------------------------------------
    // ### cuSPARSE Handle and descriptors initialization ###
    // create the test matrix on the host
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CUBLASCheck( cublasCreate(&cublasHandle) );
    CUSPARSECheck( cusparseCreate(&cusparseHandle) );
    // Create dense vectors
    CUSPARSECheck( cusparseCreateDnVec(&d_B.vec,     m, d_B.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_X.vec,     m, d_X.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_R.vec,     m, d_R.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_P.vec,   m, d_P.ptr,   CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   CUDA_R_64F) );
    
    // copy b
    auto b_vec = b.generateScalar();
    CUDACheck( cudaMemcpy(d_B.ptr, b_vec.data(), m * sizeof(double),
                           cudaMemcpyHostToDevice) );

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    cusparseSpMatDescr_t matA;
    // A
    CUSPARSECheck( cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                      d_A_columns, d_A_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) );
    
    // ### Preparation ### 
    const double Alpha = 0.75;
    size_t       bufferSizeMV;
    void*        d_bufferMV;
    double       Beta = 0.0;
    CUSPARSECheck( cusparseSpMV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &Alpha, matA, d_X.vec, &Beta, d_B.vec, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV) );
    CUDACheck( cudaMalloc(&d_bufferMV, bufferSizeMV) );

    // X0 = 0
    CUDACheck( cudaMemset(d_X.ptr, 0x0, m * sizeof(double)) );
    //--------------------------------------------------------------------------
    // ### Run CG computation ###
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CUDACheck( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    //    (b) compute R = -A * X0 + R
    CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R.vec,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) );
    //--------------------------------------------------------------------------
    // ### 2 ### P0 = R0
    CUDACheck( cudaMemcpy(d_P.ptr, d_R.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    iter = 0;
    norm = 1000;
    double nrm_R;
    CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    double delta;
    CUBLASCheck( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R.ptr, 1, &delta) );
    //--------------------------------------------------------------------------
    // ### 3 ### repeat until convergence based on max iterations and
    //           and relative residual
    // write iterative info into file
    FILE* fp = fopen("cg_info.txt", "w");
    fprintf(fp, "Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    fprintf(fp, "Iteration\tResidual\n");
    fprintf(fp, "%d\t%e\n", 0, nrm_R);
    for (int i = 0; ; i++) {
        // printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4 ### alpha = (R_i, R_i) / (A * P_i, P_i)
        //     (a) T  = A * P_i
        CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P.vec, &zero, d_T.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) );
        //     (b) denominator = (T, P_i)
        double denominator;
        CUBLASCheck( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1,
                                 &denominator) );
        //     (c) alpha = delta / denominator
        double alpha = delta / denominator;
        // PRINT_INFO(delta)
        // PRINT_INFO(denominator)
        // PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 6 ###  X_i+1 = X_i + alpha * P
        //    (a) X_i+1 = -alpha * T + X_i
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1,
                                  d_X.ptr, 1) );
        //----------------------------------------------------------------------
        // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
        //    (a) R_i+1 = -alpha * T + R_i
        double minus_alpha = -alpha;
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1,
                                  d_R.ptr, 1) );
        //----------------------------------------------------------------------
        // ### 8 ###  check ||R_i+1|| < threshold
        CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
        fprintf(fp, "%d\t%e\n", i + 1, nrm_R);
        iter++;
        if (nrm_R < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 8 ### beta = (R_i+1, R_i+1) / (R_i, R_i)
        //    (a) delta_new => (R_i+1, R_i+1)
        double delta_new;
        CUBLASCheck( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R.ptr, 1,
                                 &delta_new) );
        //    (b) beta => delta_new / delta
        double beta = delta_new / delta;
        delta       = delta_new;
        //----------------------------------------------------------------------
        // ### 9 ###  P_i+1 = R_i+1 + beta * P_i
        //    (a) copy R_i+1 in P_i
        CUDACheck( cudaMemcpy(d_P.ptr, d_R.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) );
        //    (b) P_i+1 = beta * P_i + R_i+1
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &beta, d_P.ptr, 1,
                                  d_P.ptr, 1) );
    }
    //--------------------------------------------------------------------------
    // printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CUDACheck( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    // R = -A * X + R
    CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) );
    // check ||R||
    CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
    // copy result
    CUDACheck( cudaMemcpy(h_X, d_X.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToHost) );
    std::vector<Scalar> xx(m);
    for (int i = 0; i < m; i++) {
        xx[i] = h_X[i];
    }
    
    norm = nrm_R;// * tolerance;
    printf("Final iterations: %d error norm = %e\n", iter, norm);
    
    //--------------------------------------------------------------------------
    x.setvalues({xx.begin(), xx.end()});

    //--------------------------------------------------------------------------
    // ### Free resources ###
    CUSPARSECheck( cusparseDestroyDnVec(d_B.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_X.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_R.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_P.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_T.vec) );
    CUSPARSECheck( cusparseDestroySpMat(matA) );
    CUSPARSECheck( cusparseDestroy(cusparseHandle) );
    CUBLASCheck( cublasDestroy(cublasHandle) );

    free(h_A_rows);
    free(h_A_columns);
    free(h_A_values);
    free(h_X);

    CUDACheck( cudaFree(d_X.ptr) );
    CUDACheck( cudaFree(d_B.ptr) );
    CUDACheck( cudaFree(d_R.ptr) );
    CUDACheck( cudaFree(d_P.ptr) );
    CUDACheck( cudaFree(d_T.ptr) );
    CUDACheck( cudaFree(d_A_values) );
    CUDACheck( cudaFree(d_A_columns) );
    CUDACheck( cudaFree(d_A_rows) );
    CUDACheck( cudaFree(d_bufferMV) );
    return ;
}

void PCG_ICC(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
{
    printf("PCG_ICC...\n");
    auto m_Mat = A.getMat();
	int num_rows = m_Mat.size(), nnz = 0;
	int m = num_rows;
	int num_offsets = m + 1;

    int*    h_A_rows    = (int*)    malloc(num_offsets * sizeof(int));
    h_A_rows[0] = 0;
	for (int i = 0; i < m_Mat.size(); i++) {
		for (int j = 0; j < m_Mat[i].size(); j++) {
			if (m_Mat[i][j].second != 0) {
				nnz++;
			}
		}
        h_A_rows[i + 1] = nnz;
	}
	
    int*    h_A_columns = (int*)    malloc(nnz * sizeof(int));
    double* h_A_values  = (double*) malloc(nnz * sizeof(double));
    double* h_X         = (double*) malloc(m * sizeof(double));
	// from ellpack to csr
    int cnt = 0;
    for (int i = 0; i < m_Mat.size(); i++) {
		for (int j = 0; j < m_Mat[i].size(); j++) {
			if (m_Mat[i][j].second != 0) {
				h_A_columns[cnt] = m_Mat[i][j].first;
                h_A_values[cnt] = m_Mat[i][j].second;
                cnt++;
			}
		}
	}
	
    for (int i = 0; i < num_rows; i++)
        h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_L_values;
    // R_aux = z
    Vec     d_B, d_X, d_R, d_R_aux, d_P, d_T, d_tmp;

    // allocate device memory for CSR matrices
    CUDACheck( cudaMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) );
    CUDACheck( cudaMalloc((void**) &d_A_columns, nnz * sizeof(int)) );
    CUDACheck( cudaMalloc((void**) &d_A_values,  nnz * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_L_values,  nnz * sizeof(double)) );

    CUDACheck( cudaMalloc((void**) &d_B.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_X.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_R.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_R_aux.ptr, m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_P.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_T.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_tmp.ptr,   m * sizeof(double)) );

    // copy the CSR matrices and vectors into device memory
    CUDACheck( cudaMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_L_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_X.ptr, h_X, m * sizeof(double),
                           cudaMemcpyHostToDevice) );
    //--------------------------------------------------------------------------
    // ### cuSPARSE Handle and descriptors initialization ###
    // create the test matrix on the host
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CUBLASCheck( cublasCreate(&cublasHandle) );
    CUSPARSECheck( cusparseCreate(&cusparseHandle) );
    // Create dense vectors
    CUSPARSECheck( cusparseCreateDnVec(&d_B.vec,     m, d_B.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_X.vec,     m, d_X.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_R.vec,     m, d_R.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_R_aux.vec, m, d_R_aux.ptr,
                                        CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_P.vec,   m, d_P.ptr,   CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_64F) );

    // copy b
    auto b_vec = b.generateScalar();
    CUDACheck( cudaMemcpy(d_B.ptr, b_vec.data(), m * sizeof(double),
                           cudaMemcpyHostToDevice) );

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    cusparseSpMatDescr_t matA, matL;
    int*                 d_L_rows      = d_A_rows;
    int*                 d_L_columns   = d_A_columns;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CUSPARSECheck( cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                      d_A_columns, d_A_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) );
    // L
    CUSPARSECheck( cusparseCreateCsr(&matL, m, m, nnz, d_L_rows,
                                      d_L_columns, d_L_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) );
    CUSPARSECheck( cusparseSpMatSetAttribute(matL,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)) );
    CUSPARSECheck( cusparseSpMatSetAttribute(matL,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)) );
    //--------------------------------------------------------------------------
    // ### Preparation ### b = A * X
    // from here
    const double Alpha = 0.75;
    size_t       bufferSizeMV;
    void*        d_bufferMV;
    double       Beta = 0.0;
    CUSPARSECheck( cusparseSpMV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &Alpha, matA, d_X.vec, &Beta, d_B.vec, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV) );
    CUDACheck( cudaMalloc(&d_bufferMV, bufferSizeMV) );

    // CUSPARSECheck( cusparseSpMV(
    //                     cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                     &Alpha, matA, d_X.vec, &Beta, d_B.vec, CUDA_R_64F,
    //                     CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) );
    
    // to here, maybe needed deletion or modifaction

    // X0 = 0
    CUDACheck( cudaMemset(d_X.ptr, 0x0, m * sizeof(double)) );
    //--------------------------------------------------------------------------
    // Perform Incomplete-Cholesky factorization of A (csric0) -> L, L^T
    cusparseMatDescr_t descrM;
    csric02Info_t      infoM        = NULL;
    int                bufferSizeIC = 0;
    void*              d_bufferIC;
    CUSPARSECheck( cusparseCreateMatDescr(&descrM) );
    CUSPARSECheck( cusparseSetMatIndexBase(descrM, baseIdx) );
    CUSPARSECheck( cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL) );
    CUSPARSECheck( cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER) );
    CUSPARSECheck( cusparseSetMatDiagType(descrM,
                                           CUSPARSE_DIAG_TYPE_NON_UNIT) );
    CUSPARSECheck( cusparseCreateCsric02Info(&infoM) );

    CUSPARSECheck( cusparseDcsric02_bufferSize(
                        cusparseHandle, m, nnz, descrM, d_L_values,
                        d_A_rows, d_A_columns, infoM, &bufferSizeIC) );
    CUDACheck( cudaMalloc(&d_bufferIC, bufferSizeIC) );
    CUSPARSECheck( cusparseDcsric02_analysis(
                        cusparseHandle, m, nnz, descrM, d_L_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC) );
    int structural_zero;
    CUSPARSECheck( cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                               &structural_zero) );
    // M = L * L^T
    CUSPARSECheck( cusparseDcsric02(
                        cusparseHandle, m, nnz, descrM, d_L_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC) );
    // Find numerical zero
    int numerical_zero;
    CUSPARSECheck( cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                               &numerical_zero) );

    CUSPARSECheck( cusparseDestroyCsric02Info(infoM) );
    CUSPARSECheck( cusparseDestroyMatDescr(descrM) );
    CUDACheck( cudaFree(d_bufferIC) );
    //--------------------------------------------------------------------------
    // ### Run CG computation ###
    // printf("CG loop:\n");
    // gpu_CG(cublasHandle, cusparseHandle, m,
    //        matA, matL, d_B, d_X, d_R, d_R_aux, d_P, d_T,
    //        d_tmp, d_bufferMV, maxIterations, tolerance);

    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CUDACheck( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    //    (b) compute R = -A * X0 + R
    CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R.vec,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) );
    //--------------------------------------------------------------------------
    // ### 2 ### R_i_aux = L^-1 L^-T R_i
    size_t              bufferSizeL, bufferSizeLT;
    void*               d_bufferL, *d_bufferLT;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
    //    (a) L^-T tmp => R_i_aux    (triangular solver)
    CUSPARSECheck( cusparseSpSV_createDescr(&spsvDescrLT) );
    CUSPARSECheck( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT) );
    CUDACheck( cudaMalloc(&d_bufferLT, bufferSizeLT) );
    CUSPARSECheck( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, d_bufferLT) );
    CUDACheck( cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)) );
    CUSPARSECheck( cusparseSpSV_solve(
                        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT) );

    //    (b) L^-T R_i => tmp    (triangular solver)
    CUSPARSECheck( cusparseSpSV_createDescr(&spsvDescrL) );
    CUSPARSECheck( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) );
    CUDACheck( cudaMalloc(&d_bufferL, bufferSizeL) );
    CUSPARSECheck( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) );
    CUDACheck( cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)) );
    CUSPARSECheck( cusparseSpSV_solve(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL) );
    //--------------------------------------------------------------------------
    // ### 3 ### P0 = R0_aux
    CUDACheck( cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    double delta;
    CUBLASCheck( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R.ptr, 1, &delta) );
    //--------------------------------------------------------------------------
    // ### 4 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 0; i < limit; i++) {
        // printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
        //     (a) T  = A * P_i
        CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P.vec, &zero, d_T.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) );
        //     (b) denominator = (T, P_i)
        double denominator;
        CUBLASCheck( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1,
                                 &denominator) );
        //     (c) alpha = delta / denominator
        double alpha = delta / denominator;
        // PRINT_INFO(delta)
        // PRINT_INFO(denominator)
        // PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 6 ###  X_i+1 = X_i + alpha * P
        //    (a) X_i+1 = -alpha * T + X_i
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1,
                                  d_X.ptr, 1) );
        //----------------------------------------------------------------------
        // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
        //    (a) R_i+1 = -alpha * T + R_i
        double minus_alpha = -alpha;
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1,
                                  d_R.ptr, 1) );
        //----------------------------------------------------------------------
        // ### 8 ###  check ||R_i+1|| < threshold
        CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
        // PRINT_INFO(nrm_R)
        iter++;
        if (nrm_R < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
        //    (a) L^-T R_i+1 => tmp    (triangular solver)
        CUDACheck( cudaMemset(d_tmp.ptr,   0x0, m * sizeof(double)) );
        CUDACheck( cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)) );
        CUSPARSECheck( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matL, d_R.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) );
        //    (b) L^-T tmp => R_aux_i+1    (triangular solver)
        CUSPARSECheck( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_TRANSPOSE,
                                           &one, matL, d_tmp.vec,
                                           d_R_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrLT) );
        //----------------------------------------------------------------------
        // ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
        //    (a) delta_new => (R_i+1, R_aux_i+1)
        double delta_new;
        CUBLASCheck( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1,
                                 &delta_new) );
        //    (b) beta => delta_new / delta
        double beta = delta_new / delta;
        // PRINT_INFO(delta_new)
        // PRINT_INFO(beta)
        delta       = delta_new;
        //----------------------------------------------------------------------
        // ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
        //    (a) copy R_aux_i+1 in P_i
        CUDACheck( cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) );
        //    (b) P_i+1 = beta * P_i + R_aux_i+1
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &beta, d_P.ptr, 1,
                                  d_P.ptr, 1) );
    }
    //--------------------------------------------------------------------------
    // printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CUDACheck( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    // R = -A * X + R
    CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) );
    // check ||R||
    CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
    // copy result
    CUDACheck( cudaMemcpy(h_X, d_X.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToHost) );
    std::vector<Scalar> xx(m);
    for (int i = 0; i < m; i++) {
        xx[i] = h_X[i];
    }
    
    norm = nrm_R * tolerance;
    // printf("Final iterations: %d error norm = %e\n", iter, norm);
    
    //--------------------------------------------------------------------------
    CUSPARSECheck( cusparseSpSV_destroyDescr(spsvDescrL) );
    CUSPARSECheck( cusparseSpSV_destroyDescr(spsvDescrLT) );
    CUDACheck( cudaFree(d_bufferL) );
    CUDACheck( cudaFree(d_bufferLT) );

    x.setvalues({xx.begin(), xx.end()});

    //--------------------------------------------------------------------------
    // ### Free resources ###
    CUSPARSECheck( cusparseDestroyDnVec(d_B.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_X.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_R.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_R_aux.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_P.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_T.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_tmp.vec) );
    CUSPARSECheck( cusparseDestroySpMat(matA) );
    CUSPARSECheck( cusparseDestroySpMat(matL) );
    CUSPARSECheck( cusparseDestroy(cusparseHandle) );
    CUBLASCheck( cublasDestroy(cublasHandle) );

    free(h_A_rows);
    free(h_A_columns);
    free(h_A_values);
    free(h_X);

    CUDACheck( cudaFree(d_X.ptr) );
    CUDACheck( cudaFree(d_B.ptr) );
    CUDACheck( cudaFree(d_R.ptr) );
    CUDACheck( cudaFree(d_R_aux.ptr) );
    CUDACheck( cudaFree(d_P.ptr) );
    CUDACheck( cudaFree(d_T.ptr) );
    CUDACheck( cudaFree(d_tmp.ptr) );
    CUDACheck( cudaFree(d_A_values) );
    CUDACheck( cudaFree(d_A_columns) );
    CUDACheck( cudaFree(d_A_rows) );
    CUDACheck( cudaFree(d_L_values) );
    CUDACheck( cudaFree(d_bufferMV) );
    return ;
}

void BiCGSTAB(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
{
    printf("BiCGSTAB...\n");
    auto m_Mat = A.getMat();
	int num_rows = m_Mat.size(), nnz = 0;
	int m = num_rows;
	int num_offsets = m + 1;

    int*    h_A_rows    = (int*)    malloc(num_offsets * sizeof(int));
    h_A_rows[0] = 0;
	for (int i = 0; i < m_Mat.size(); i++) {
		for (int j = 0; j < m_Mat[i].size(); j++) {
			if (m_Mat[i][j].second != 0) {
				nnz++;
			}
		}
        h_A_rows[i + 1] = nnz;
	}
	
    int*    h_A_columns = (int*)    malloc(nnz * sizeof(int));
    double* h_A_values  = (double*) malloc(nnz * sizeof(double));
    double* h_X         = (double*) malloc(m * sizeof(double));
	// from ellpack to csr
    int cnt = 0;
    for (int i = 0; i < m_Mat.size(); i++) {
		for (int j = 0; j < m_Mat[i].size(); j++) {
			if (m_Mat[i][j].second != 0) {
				h_A_columns[cnt] = m_Mat[i][j].first;
                h_A_values[cnt] = m_Mat[i][j].second;
                cnt++;
            }
		}
	}

    // printf("Testing BiCGStab\n");
    for (int i = 0; i < num_rows; i++)
        h_X[i] = 0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_M_values;
    Vec     d_B, d_X, d_R, d_R0, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T, d_tmp;

    // allocate device memory for CSR matrices
    CUDACheck( cudaMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) );
    CUDACheck( cudaMalloc((void**) &d_A_columns, nnz * sizeof(int)) );
    CUDACheck( cudaMalloc((void**) &d_A_values,  nnz * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_M_values,  nnz * sizeof(double)) );

    CUDACheck( cudaMalloc((void**) &d_B.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_X.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_R.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_R0.ptr,    m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_P.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_P_aux.ptr, m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_S.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_S_aux.ptr, m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_V.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_T.ptr,     m * sizeof(double)) );
    CUDACheck( cudaMalloc((void**) &d_tmp.ptr,   m * sizeof(double)) );

    // copy the CSR matrices and vectors into device memory
    CUDACheck( cudaMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_M_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) );
    CUDACheck( cudaMemcpy(d_X.ptr, h_X, m * sizeof(double),
                           cudaMemcpyHostToDevice) );
    //--------------------------------------------------------------------------
    // ### cuSPARSE Handle and descriptors initialization ###
    // create the test matrix on the host
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CUBLASCheck( cublasCreate(&cublasHandle) );
    CUSPARSECheck( cusparseCreate(&cusparseHandle) );
    // Create dense vectors
    CUSPARSECheck( cusparseCreateDnVec(&d_B.vec,     m, d_B.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_X.vec,     m, d_X.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_R.vec,     m, d_R.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_R0.vec,    m, d_R0.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_P.vec,     m, d_P.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_P_aux.vec, m, d_P_aux.ptr,
                                        CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_S.vec,     m, d_S.ptr, CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_S_aux.vec, m, d_S_aux.ptr,
                                        CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_V.vec,   m, d_V.ptr,   CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   CUDA_R_64F) );
    CUSPARSECheck( cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_64F) );

    // copy b
    auto b_vec = b.generateScalar();
    CUDACheck( cudaMemcpy(d_B.ptr, b_vec.data(), m * sizeof(double),
                           cudaMemcpyHostToDevice) );

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    // IMPORTANT: Upper/Lower triangular decompositions of A
    //            (matM_lower, matM_upper) must use two distinct descriptors
    cusparseSpMatDescr_t matA, matM_lower, matM_upper;
    cusparseMatDescr_t   matLU;
    int*                 d_M_rows      = d_A_rows;
    int*                 d_M_columns   = d_A_columns;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t   fill_upper    = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CUSPARSECheck( cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                      d_A_columns, d_A_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) );
    // M_lower
    CUSPARSECheck( cusparseCreateCsr(&matM_lower, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) );
    CUSPARSECheck( cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)) );
    CUSPARSECheck( cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unit, sizeof(diag_unit)) );
    // M_upper
    CUSPARSECheck( cusparseCreateCsr(&matM_upper, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) );
    CUSPARSECheck( cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_upper, sizeof(fill_upper)) );
    CUSPARSECheck( cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)) );
    //--------------------------------------------------------------------------
    // ### Preparation ### b = A * X
    const double Alpha = 0.75;
    size_t bufferSizeMV;
    void*  d_bufferMV;
    double beta = 0.0;
    CUSPARSECheck( cusparseSpMV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &Alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV) );
    CUDACheck( cudaMalloc(&d_bufferMV, bufferSizeMV) );

    // CUSPARSECheck( cusparseSpMV(
    //                     cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                     &Alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
    //                     CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) );
    // X0 = 0
    CUDACheck( cudaMemset(d_X.ptr, 0x0, m * sizeof(double)) );
    //--------------------------------------------------------------------------
    // Perform Incomplete-LU factorization of A (csrilu0) -> M_lower, M_upper
    csrilu02Info_t infoM        = NULL;
    int            bufferSizeLU = 0;
    void*          d_bufferLU;
    CUSPARSECheck( cusparseCreateMatDescr(&matLU) );
    CUSPARSECheck( cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL) );
    CUSPARSECheck( cusparseSetMatIndexBase(matLU, baseIdx) );
    CUSPARSECheck( cusparseCreateCsrilu02Info(&infoM) );

    CUSPARSECheck( cusparseDcsrilu02_bufferSize(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM, &bufferSizeLU) );
    CUDACheck( cudaMalloc(&d_bufferLU, bufferSizeLU) );
    CUSPARSECheck( cusparseDcsrilu02_analysis(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) );
    int structural_zero;
    CUSPARSECheck( cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM,
                                                &structural_zero) );
    // M = L * U
    CUSPARSECheck( cusparseDcsrilu02(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) );
    // Find numerical zero
    int numerical_zero;
    CUSPARSECheck( cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM,
                                                &numerical_zero) );

    CUSPARSECheck( cusparseDestroyCsrilu02Info(infoM) );
    CUSPARSECheck( cusparseDestroyMatDescr(matLU) );
    CUDACheck( cudaFree(d_bufferLU) );
    //--------------------------------------------------------------------------
    // ### Run BiCGStab computation ###
    printf("BiCGStab loop:\n");
    // gpu_BiCGStab(cublasHandle, cusparseHandle, m,
    //              matA, matM_lower, matM_upper,
    //              d_B, d_X, d_R0, d_R, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T,
    //              d_tmp, d_bufferMV, maxIterations, tolerance);


    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // Create opaque data structures that holds analysis data between calls
    double              coeff_tmp;
    size_t              bufferSizeL, bufferSizeU;
    void*               d_bufferL, *d_bufferU;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    CUSPARSECheck( cusparseSpSV_createDescr(&spsvDescrL) );
    CUSPARSECheck( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) );
    CUDACheck( cudaMalloc(&d_bufferL, bufferSizeL) );
    CUSPARSECheck( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) );

    // Calculate UPPER buffersize
    CUSPARSECheck( cusparseSpSV_createDescr(&spsvDescrU) );
    CUSPARSECheck( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        &bufferSizeU) );
    CUDACheck( cudaMalloc(&d_bufferU, bufferSizeU) );
    CUSPARSECheck( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        d_bufferU) );
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CUDACheck( cudaMemcpy(d_R0.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    //    (b) compute R = -A * X0 + R
    CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R0.vec,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) );
    //--------------------------------------------------------------------------
    double alpha, delta, delta_prev, omega;
    CUBLASCheck( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_R0.ptr, 1,
                             &delta) );
    delta_prev = delta;
    // R = R0
    CUDACheck( cudaMemcpy(d_R.ptr, d_R0.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R0.ptr, 1, &nrm_R) );
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 1; i <= limit; i++) {
        // printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        CUDACheck( cudaMemcpy(d_P.ptr, d_R.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) );
        if (i > 1) {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            CUBLASCheck( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_R.ptr, 1,
                                     &delta) );
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = (delta / delta_prev) * (alpha / omega);
            delta_prev  = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            double minus_omega = -omega;
            CUBLASCheck( cublasDaxpy(cublasHandle, m, &minus_omega, d_V.ptr, 1,
                                      d_P.ptr, 1) );
            //    (b) P = beta * P
            CUBLASCheck( cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1) );
            //    (c) P = R + P
            CUBLASCheck( cublasDaxpy(cublasHandle, m, &one, d_R.ptr, 1,
                                      d_P.ptr, 1) );
        }
        //----------------------------------------------------------------------
        // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        //    (a) M_L^-1 P_i => tmp    (triangular solver)
        CUDACheck( cudaMemset(d_tmp.ptr,   0x0, m * sizeof(double)) );
        CUDACheck( cudaMemset(d_P_aux.ptr, 0x0, m * sizeof(double)) );
        CUSPARSECheck( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_P.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) );
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        CUSPARSECheck( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_P_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU) );
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P_aux.vec, &zero, d_V.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) );
        //    (b) denominator = R'0 * V
        double denominator;
        CUBLASCheck( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_V.ptr, 1,
                                 &denominator) );
        alpha = delta / denominator;
        // PRINT_INFO(delta)
        // PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &alpha, d_P_aux.ptr, 1,
                                  d_X.ptr, 1) );
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        CUDACheck( cudaMemcpy(d_S.ptr, d_R.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) );
        //    (b) S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &minus_alpha, d_V.ptr, 1,
                                  d_S.ptr, 1) );
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        double nrm_S;
        CUBLASCheck( cublasDnrm2(cublasHandle, m, d_S.ptr, 1, &nrm_S) );
        // PRINT_INFO(nrm_S)
        iter++;
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        //    (a) M_L^-1 S => tmp    (triangular solver)
        cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double));
        cudaMemset(d_S_aux.ptr, 0x0, m * sizeof(double));
        CUSPARSECheck( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_S.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) );
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        CUSPARSECheck( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_S_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU) );
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_S_aux.vec, &zero, d_T.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) );
        //    (b) omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        CUBLASCheck( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_S.ptr, 1,
                                 &omega_num) );
        //    (c) omega_den = (A * S_aux, A * S_aux)
        CUBLASCheck( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_T.ptr, 1,
                                 &omega_den) );
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        // PRINT_INFO(omega)
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &omega, d_S_aux.ptr, 1,
                                  d_X.ptr, 1) );
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        CUDACheck( cudaMemcpy(d_R.ptr, d_S.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) );
        //    (a) R_i+1 = -omega * T + R
        double minus_omega = -omega;
        CUBLASCheck( cublasDaxpy(cublasHandle, m, &minus_omega, d_T.ptr, 1,
                                  d_R.ptr, 1) );
       // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
        // PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
    }
    //--------------------------------------------------------------------------
    printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CUDACheck( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) );
    // R = -A * X + R
    CUSPARSECheck( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) );
    // check ||R||
    CUBLASCheck( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) );
    CUDACheck( cudaMemcpy(h_X, d_X.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToHost) );
    std::vector<Scalar> xx(m);
    for (int i = 0; i < m; i++)
    {
        xx[i] = h_X[i];
    }
    
    printf("Final iterations: %d error norm = %e\n", iter, nrm_R);
    //--------------------------------------------------------------------------
    CUSPARSECheck( cusparseSpSV_destroyDescr(spsvDescrL) );
    CUSPARSECheck( cusparseSpSV_destroyDescr(spsvDescrU) );
    CUDACheck( cudaFree(d_bufferL) );
    CUDACheck( cudaFree(d_bufferU) );

    x.setvalues({xx.begin(), xx.end()});

    //--------------------------------------------------------------------------
    // ### Free resources ###
    CUSPARSECheck( cusparseDestroyDnVec(d_B.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_X.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_R.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_R0.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_P.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_P_aux.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_S.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_S_aux.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_V.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_T.vec) );
    CUSPARSECheck( cusparseDestroyDnVec(d_tmp.vec) );
    CUSPARSECheck( cusparseDestroySpMat(matA) );
    CUSPARSECheck( cusparseDestroySpMat(matM_lower) );
    CUSPARSECheck( cusparseDestroySpMat(matM_upper) );
    CUSPARSECheck( cusparseDestroy(cusparseHandle) );
    CUBLASCheck( cublasDestroy(cublasHandle) );

    free(h_A_rows);
    free(h_A_columns);
    free(h_A_values);
    free(h_X);

    CUDACheck( cudaFree(d_X.ptr) );
    CUDACheck( cudaFree(d_B.ptr) );
    CUDACheck( cudaFree(d_R.ptr) );
    CUDACheck( cudaFree(d_R0.ptr) );
    CUDACheck( cudaFree(d_P.ptr) );
    CUDACheck( cudaFree(d_P_aux.ptr) );
    CUDACheck( cudaFree(d_S.ptr) );
    CUDACheck( cudaFree(d_S_aux.ptr) );
    CUDACheck( cudaFree(d_V.ptr) );
    CUDACheck( cudaFree(d_T.ptr) );
    CUDACheck( cudaFree(d_tmp.ptr) );
    CUDACheck( cudaFree(d_A_values) );
    CUDACheck( cudaFree(d_A_columns) );
    CUDACheck( cudaFree(d_A_rows) );
    CUDACheck( cudaFree(d_M_values) );
    CUDACheck( cudaFree(d_bufferMV) );
    return ;
}