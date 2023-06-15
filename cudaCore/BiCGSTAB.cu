#include "BICGSTAB.cuh"
#include<vector>
#include<iostream>

CudaVector::~CudaVector()
{
}

void CudaVector::AllocateData(const Vector& vec)
{
	auto v = vec.generateScalar();
	thrust::copy(v.begin(),v.end(),values.begin());
}

CudaSPVector::~CudaSPVector()
{
}

void CudaSPVector::AllocateData(const Vector& vector)
{
	std::vector<idxType> ii;
	std::vector<Scalar> vv;
	for (int i = 0; i < vector.size(); i++)
	{
		if(vector[i] != 0)
			ii.push_back(i),vv.push_back(vector[i]);
	}
	
}

CudaSPMatrix::~CudaSPMatrix()
{
	// for (int i = 0; i < row; i++)
	// {
	// 	cudaFree(matrix[i]);
	// }
	// cudaFree(dev_matrix);
	// cudaFree(preA);
	// delete[] matrix;
}

void CudaSPMatrix::AllocateData(const SymetrixSparseMatrix& mat)
{
	// matrix = new IndexValue*[row];
	// auto prea = new int[row];
	// cudaMalloc((void**)&preA, sizeof(int) * row);
	std::vector<idxType> co(MaxCol * row),ro(row);
	std::vector<Scalar> va(MaxCol * row);

	for (int i = 0; i < row; i++)
	{
		// std::vector<IndexValue> vv;
		for (auto& kv:mat.getRow(i))
		{
			if(kv.second != 0)
			{
				// vv.push_back({(int)kv.first,kv.second});
				co[ro[i] * row + i] = kv.first;
				va[row * ro[i] + i] = kv.second,ro[i]++;
			}
		}
		// prea[i] = vv.size();
		// cudaMalloc((void**)&matrix[i], sizeof(IndexValue) * vv.size());
		// cudaMemcpy(matrix[i], vv.data(), sizeof(IndexValue) * vv.size(),cudaMemcpyHostToDevice);
	}
	colume = {co.begin(),co.end()};
	value = {va.begin(),va.end()};
	// cudaMemcpy(preA, prea, sizeof(int) * row,cudaMemcpyHostToDevice);
	// delete []prea;
	// cudaMalloc((void**)&dev_matrix, sizeof(IndexValue*) * row);
	// cudaMemcpy(dev_matrix, matrix, sizeof(IndexValue*) * row,cudaMemcpyHostToDevice);
}

__global__ void computeP(Scalar* p,Scalar* r,Scalar* v,int length,Scalar beta,Scalar w)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	p[id] = r[id] + beta * (p[id] - w * v[id]);
}


__global__ void computeP_PCG(Scalar* p,Scalar* z,int length,Scalar beta)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	p[id] = z[id] + beta * p[id];
}

__global__ void MatrixMultVector(Scalar* v1,Scalar* v2,IndexValue** matrix,int* preA,int length)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	v1[id] = 0;
	for (int i = 0; i < preA[id]; i++)
	{
		v1[id] += matrix[id][i].value * v2[matrix[id][i].colid];
	}
}

__global__ void computeS(Scalar* s,Scalar* r,Scalar* v,int length,Scalar alpha)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	s[id] = r[id] - alpha * v[id];
}

__global__ void computeX(Scalar* x,Scalar* p,Scalar* s,int length,Scalar alpha,double w)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	x[id] = x[id] + alpha * p[id] + w * s[id];
}

__global__ void MatrixMultVector_ELL(Scalar* v1,Scalar* v2,idxType* col,Scalar* values,int row,int batch)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	// if(id >= length)
	// 	return;
	id *= batch;
	int len = min(row,id + batch);
	// #pragma unroll
	for(;id < len;id++)
	{
		v1[id] = 0;
		for (int i = 0; i < MaxCol; i++)
		{
			if(values[i * row + id] == 0)
				break;
			v1[id] += values[i * row + id] * v2[col[i * row + id]];
		}
	}
}

__global__ void computeP_CG(Scalar* p,Scalar* r,int length,Scalar beta,int batch)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	// if(id >= length)
	// 	return;
	id *= batch;
	int len = min(length,id + batch);
	// float3* pp = reinterpret_cast<float3*>(p);
	// float3* rr = reinterpret_cast<float3*>(r);
	for (; id < len; id+=1)
	{
		p[id] = r[id] + beta * p[id];
		// pp[id].x = rr[id].x + beta * pp[id].x;
		// pp[id].y = rr[id].y + beta * pp[id].y;
		// pp[id].z = rr[id].z + beta * pp[id].z;
		// float3 rrr = rr[id];
		// pp[id].x = rrr.x + beta * pp[id].x;
		// pp[id].y = rrr.y + beta * pp[id].y;
		// pp[id].z = rrr.z + beta * pp[id].z;
		// p[id + 1] = r[id + 1] + beta * p[id + 1];
		// p[id + 2] = r[id + 2] + beta * p[id + 2];
	}
}

__global__ void computeX_CG(Scalar* x,Scalar* p,int length,Scalar alpha,int batch)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	// if(id >= length)
	// 	return;
	id *= batch;
	int len = min(length,id + batch);
	for (; id < len; id++)
	{
		x[id] = x[id] + alpha * p[id] ;
	}
}

__global__ void computeR_CG(Scalar* r,Scalar* Ap,int length,Scalar alpha,int batch)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	// if(id >= length)
	// 	return;
	id *= batch;
	int len = min(length,id + batch);
	for (; id < len; id++)
	{
		r[id] = r[id] - alpha * Ap[id];
	}
}

// __global__ void computeX_R_CG(Scalar* x,Scalar* p,Scalar* r,Scalar* Ap,int length,double alpha,int batch)
__global__ void computeX_R_CG(Scalar* x,Scalar* p,Scalar* r,Scalar* Ap,int length,Scalar alpha,int batch)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	// if(id >= length)
	// 	return;
	id *= batch;
	int len = min(length,id + batch);
	// float3* xx = reinterpret_cast<float3*>(x);
	// float3* rr = reinterpret_cast<float3*>(r);
	// float3* pp = reinterpret_cast<float3*>(p);
	// float3* AApp = reinterpret_cast<float3*>(Ap);
	for (; id < len; id += 1)
	{
		// float3 xxx = xx[id];
		// float3 rrr = rr[id];
		// float3 ppp = pp[id];
		// float3 AAAppp = AApp[id];

		// xxx.x += alpha * ppp.x;
		// xxx.y += alpha * ppp.y;
		// xxx.z += alpha * ppp.z;

		// rrr.x -= alpha * AAAppp.x;
		// rrr.y -= alpha * AAAppp.y;
		// rrr.z -= alpha * AAAppp.z;

		// xx[id] = xxx;
		// rr[id] = rrr;

		x[id] = x[id] + alpha * p[id];
		r[id] = r[id] - alpha * Ap[id];

		// x[id + 1] = x[id + 1] + alpha * p[id + 1];
		// r[id + 1] = r[id + 1] - alpha * Ap[id + 1];

		// x[id + 2] = x[id + 2] + alpha * p[id + 2];
		// r[id + 2] = r[id + 2] - alpha * Ap[id + 2];
	}
}

__global__ void computeX_PCG(Scalar* x,Scalar* p,int length,Scalar alpha)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	x[id] = x[id] + alpha * p[id] ;
}

__global__ void computeR(Scalar* r,Scalar* s,Scalar* t,int length,Scalar w)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	r[id] = s[id] - w * t[id];
}

__global__ void computeR_PCG(Scalar* r,Scalar* w,int length,Scalar alpha)
{
	auto id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= length)
		return;
	r[id] = r[id] - alpha * w[id];
}

void SolveTriL(const SymetrixSparseMatrix& m,thrust::host_vector<Scalar>& x,const thrust::host_vector<Scalar>& b)
{
    for (int i = 0; i < m.get_row(); i++)
    {
        double rest = b[i];
        for(auto& col:m.getRow(i))
        // for(int i = 0;i < 81;i++)
        {
            // rest -= 88 * x[0];
            if(col.first == i)
                break;
            rest -= col.second * x[col.first];
        }
        x[i] = rest / m.index(i,i);
    }
}

void SolveTriU(const SymetrixSparseMatrix& m,thrust::host_vector<Scalar>& x,const thrust::host_vector<Scalar>& b)
{
    for (int i = m.get_row() - 1; i >= 0; i--)
    {
        double rest = b[i];
        for(auto& col:m.getRow(i))
        // for(int i = 0;i < 81;i++)
        {
            // rest -= 88 * x[0];
            if(col.first == i)
                continue;
            rest -= col.second * x[col.first];
        }
        x[i] = rest / m.index(i,i);
    }
}

void CG(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
{
	const int batch = 1;
	const int ts = 32 * batch;
	int bs = (A.get_row() / ts * ts == A.get_row()) ?  A.get_row() / ts : A.get_row() / ts + 1;
	// int bs = 32;
	dim3 blockSize(bs);
	dim3 blockSize2(bs / 3);
	dim3 threadSize(ts / batch);
	// int mult = bs * ts;
	// int batch = (A.get_row() / mult * mult == A.get_row()) ?  A.get_row() / mult : A.get_row() / mult + 1;
	Scalar alpha = 0.0,rr0,rr1,beta = 0.0;

	CudaSPMatrix cspm(A.get_row(),A.get_col(),A);

	thrust::device_vector<Scalar> r(b.begin(),b.end()),xx(b.size()),p = r,Ap(b.size()),temp(b.size());

	iter = 0;
	norm = 1000;
	double normb = b.norm1();
	thrust::transform(thrust::device,r.begin(),r.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
	rr1 = thrust::reduce(thrust::device,temp.begin(),temp.end());

	// std::vector<Scalar> tempp;

	while(iter < limit && norm > tolerance * normb)
	{
		MatrixMultVector_ELL<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&Ap[0]),thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&(cspm.colume[0])),thrust::raw_pointer_cast(&(cspm.value[0])),Ap.size(),batch);
		// tempp = {Ap.begin(),Ap.end()};
		thrust::transform(thrust::device,p.begin(),p.end(),Ap.begin(),temp.begin(),thrust::multiplies<Scalar>());
		alpha = rr1 / thrust::reduce(thrust::device,temp.begin(),temp.end());

		computeX_R_CG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&xx[0]),thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&r[0]),thrust::raw_pointer_cast(&Ap[0]),r.size(),alpha,batch);
		// tempp = {r.begin(),r.end()};

		rr0 = rr1;
		thrust::transform(thrust::device,r.begin(),r.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
		rr1 = thrust::reduce(thrust::device,temp.begin(),temp.end());
		beta = rr1 / rr0;
		computeP_CG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&r[0]),p.size(),beta,batch);
		// tempp = {p.begin(),p.end()};
		
		iter++;
		norm = std::sqrt(rr1);
	}
	x.setvalues({xx.begin(),xx.end()});
}

void PCG_ICC(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
{
	const int batch = 1;
	const int ts = 32 * batch;
	int bs = (A.get_row() / ts * ts == A.get_row()) ?  A.get_row() / ts : A.get_row() / ts + 1;
	dim3 blockSize(bs);
	dim3 threadSize(ts);
	Scalar alpha = 0.0,rr0,rr1,beta = 0.0;
	// auto precon = A.ichol().inverse_lowertri();
	auto precon = A.ichol();
	auto preconT = precon.transpose();
	CudaSPMatrix cspm(A.get_row(),A.get_col(),A);
	// CudaSPMatrix prec(precon.get_row(),precon.get_col(),precon);
	// CudaSPMatrix precT(preconT.get_row(),preconT.get_col(),preconT);

	thrust::device_vector<Scalar> r(b.begin(),b.end()),xx(b.size()),p(b.size()),Ap(b.size()),temp(b.size()),
	y(b.size()),z(b.size()),w(b.size()),lastr(b.size());

    auto startT = std::chrono::high_resolution_clock::now();
	std::cout << "start!" << std::endl;

	// MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&y[0]),thrust::raw_pointer_cast(&r[0]),prec.dev_matrix,prec.preA,y.size());
	// MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&z[0]),thrust::raw_pointer_cast(&y[0]),precT.dev_matrix,precT.preA,z.size());
	thrust::host_vector<Scalar> r_host(b.begin(),b.end()),y_host(b.size()),z_host(b.size());
	SolveTriL(precon,y_host,r_host);
	SolveTriU(preconT,z_host,y_host);
	z = {z_host.begin(),z_host.end()};
	p = z;
	std::vector<Scalar> tempp{z.begin(),z.end()};
	// std::vector<Scalar> tempp2{y.begin(),y.end()};
	iter = 0;
	thrust::transform(thrust::device,r.begin(),r.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
	norm = 1000;
	double normb = b.norm1();

	thrust::transform(thrust::device,r.begin(),r.end(),z.begin(),temp.begin(),thrust::multiplies<Scalar>());
	rr0 = thrust::reduce(thrust::device,temp.begin(),temp.end());

	while(iter < limit && norm > tolerance * normb)
	{
		// MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&w[0]),thrust::raw_pointer_cast(&p[0]),cspm.dev_matrix,cspm.preA,Ap.size());
		MatrixMultVector_ELL<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&Ap[0]),thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&(cspm.colume[0])),thrust::raw_pointer_cast(&(cspm.value[0])),Ap.size(),batch);

		// std::vector<Scalar> tempp{w.begin(),w.end()};
		thrust::transform(thrust::device,p.begin(),p.end(),Ap.begin(),temp.begin(),thrust::multiplies<Scalar>());
		alpha = rr0 / thrust::reduce(thrust::device,temp.begin(),temp.end());

		computeX_PCG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&xx[0]),thrust::raw_pointer_cast(&p[0]),xx.size(),alpha);
		// tempp = {xx.begin(),xx.end()};

		// lastr = r;
		computeR_PCG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&r[0]),thrust::raw_pointer_cast(&Ap[0]),r.size(),alpha);
		// tempp = {r.begin(),r.end()};

		// MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&y[0]),thrust::raw_pointer_cast(&r[0]),prec.dev_matrix,prec.preA,y.size());
		// MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&z[0]),thrust::raw_pointer_cast(&y[0]),precT.dev_matrix,precT.preA,z.size());
		r_host= {r.begin(),r.end()};
		SolveTriL(precon,y_host,r_host);
		SolveTriU(preconT,z_host,y_host);
		z = {z_host.begin(),z_host.end()};
		// tempp = {z.begin(),z.end()};

		// thrust::transform(thrust::device,r.begin(),r.end(),lastr.begin(),temp.begin(),thrust::minus<Scalar>());
		// thrust::transform(thrust::device,temp.begin(),temp.end(),z.begin(),temp.begin(),thrust::multiplies<Scalar>());
		thrust::transform(thrust::device,r.begin(),r.end(),z.begin(),temp.begin(),thrust::multiplies<Scalar>());
		rr1 = thrust::reduce(thrust::device,temp.begin(),temp.end());

		beta = rr1 / rr0;
		// thrust::transform(thrust::device,r.begin(),r.end(),z.begin(),temp.begin(),thrust::multiplies<Scalar>());
		// rr0 = thrust::reduce(thrust::device,temp.begin(),temp.end());
		rr0 = rr1;

		computeP_PCG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&z[0]),p.size(),beta);
		// tempp = {p.begin(),p.end()};

		iter++;
		// thrust::transform(thrust::device,r.begin(),r.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
		// norm = thrust::reduce(thrust::device,temp.begin(),temp.end());
		norm = std::sqrt(rr1);
		std::cout << iter << " " << norm << std::endl;
	}
	x.setvalues({xx.begin(),xx.end()});
    auto endT = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endT - startT);
    
    printf("SOLVETIME = %.5le \n", static_cast<double>(duration.count()));
}

void PCG_SSOR(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
{
	int bs = (A.get_row() / 32 * 32 == A.get_row()) ?  A.get_row() / 32 : A.get_row() / 32 + 1;
	dim3 blockSize(bs);
	dim3 threadSize(32);
	Scalar alpha = 0.0,rr0,rr1,beta = 0.0;

	auto precon = A.SSORAI();
	auto preconT = precon.transpose();
	CudaSPMatrix cspm(A.get_row(),A.get_col(),A);
	CudaSPMatrix prec(precon.get_row(),precon.get_col(),precon);
	CudaSPMatrix precT(precon.get_row(),precon.get_col(),preconT);

	thrust::device_vector<Scalar> r(b.begin(),b.end()),xx(b.size()),p(b.size()),Ap(b.size()),temp(b.size()),z(b.size()),w(b.size()),lastr(b.size());

	MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&temp[0]),thrust::raw_pointer_cast(&r[0]),prec.dev_matrix,prec.preA,r.size());
	MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&z[0]),thrust::raw_pointer_cast(&temp[0]),precT.dev_matrix,precT.preA,z.size());
	p = z;
	std::vector<Scalar> tempp{z.begin(),z.end()};
	iter = 0;
	// thrust::transform(thrust::device,r.begin(),r.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
	norm = 1000;
	double normb = b.norm1();

	thrust::transform(thrust::device,r.begin(),r.end(),z.begin(),temp.begin(),thrust::multiplies<Scalar>());
	rr0 = thrust::reduce(thrust::device,temp.begin(),temp.end());

	while(iter < limit && norm > tolerance * normb)
	{
		MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&w[0]),thrust::raw_pointer_cast(&p[0]),cspm.dev_matrix,cspm.preA,Ap.size());

		// std::vector<Scalar> tempp{w.begin(),w.end()};
		// tempp = {w.begin(),w.end()};
		thrust::transform(thrust::device,p.begin(),p.end(),w.begin(),temp.begin(),thrust::multiplies<Scalar>());
		alpha = rr0 / thrust::reduce(thrust::device,temp.begin(),temp.end());

		computeX_PCG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&xx[0]),thrust::raw_pointer_cast(&p[0]),xx.size(),alpha);
		// tempp = {xx.begin(),xx.end()};

		// lastr = r;
		computeR_PCG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&r[0]),thrust::raw_pointer_cast(&w[0]),r.size(),alpha);
		// tempp = {r.begin(),r.end()};

		MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&temp[0]),thrust::raw_pointer_cast(&r[0]),prec.dev_matrix,prec.preA,r.size());
		MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&z[0]),thrust::raw_pointer_cast(&temp[0]),precT.dev_matrix,precT.preA,z.size());
		// tempp = {z.begin(),z.end()};


		// thrust::transform(thrust::device,r.begin(),r.end(),lastr.begin(),temp.begin(),thrust::minus<Scalar>());
		thrust::transform(thrust::device,r.begin(),r.end(),z.begin(),temp.begin(),thrust::multiplies<Scalar>());
		rr1 = thrust::reduce(thrust::device,temp.begin(),temp.end());

		beta = rr1 / rr0;
		// thrust::transform(thrust::device,r.begin(),r.end(),z.begin(),temp.begin(),thrust::multiplies<Scalar>());
		// rr0 = thrust::reduce(thrust::device,temp.begin(),temp.end());
		rr0 = rr1;

		// tempp = {p.begin(),p.end()};
		computeP_PCG<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&z[0]),p.size(),beta);

		iter++;
		thrust::transform(thrust::device,r.begin(),r.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
		norm = thrust::reduce(thrust::device,temp.begin(),temp.end());
		norm = std::sqrt(norm);
	}
	x.setvalues({xx.begin(),xx.end()});
}

void BiCGSTAB(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
{
	int bs = (A.get_row() / 32 * 32 == A.get_row()) ?  A.get_row() / 32 : A.get_row() / 32 + 1;
	dim3 blockSize(bs);
	dim3 threadSize(32);
	Scalar rho0,w,alpha,rho1;
	rho0 = w = alpha = 1.0;

	CudaSPMatrix cspm(A.get_row(),A.get_col(),A);
	thrust::device_vector<Scalar> r(b.begin(),b.end()),xx(b.size()),r_hat = r,v(b.size()),p(b.size()),s(b.size()),t(b.size()),temp(b.size());
	iter = 0;
	norm = 1000;
	double normb = b.norm1();
	while(iter < limit && norm > tolerance * normb)
	{
		thrust::transform(thrust::device,r_hat.begin(),r_hat.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
		rho1 = thrust::reduce(thrust::device,temp.begin(),temp.end());
		double beta = rho1 / rho0 * alpha / w;
		rho0 = rho1;

		computeP<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&r[0]),thrust::raw_pointer_cast(&v[0]),p.size(),beta,w);
		MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&v[0]),thrust::raw_pointer_cast(&p[0]),cspm.dev_matrix,cspm.preA,v.size());

		thrust::transform(thrust::device,r_hat.begin(),r_hat.end(),v.begin(),temp.begin(),thrust::multiplies<Scalar>());
		alpha = rho1 / thrust::reduce(thrust::device,temp.begin(),temp.end());

		computeS<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&s[0]),thrust::raw_pointer_cast(&r[0]),thrust::raw_pointer_cast(&v[0]),s.size(),alpha);
		MatrixMultVector<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&t[0]),thrust::raw_pointer_cast(&s[0]),cspm.dev_matrix,cspm.preA,t.size());


		thrust::transform(thrust::device,s.begin(),s.end(),t.begin(),temp.begin(),thrust::multiplies<Scalar>());
		w = thrust::reduce(thrust::device,temp.begin(),temp.end());
		thrust::transform(thrust::device,t.begin(),t.end(),t.begin(),temp.begin(),thrust::multiplies<Scalar>());
		w = w / thrust::reduce(thrust::device,temp.begin(),temp.end());


		// thrust::transform(thrust::device,r_hat.begin(),r_hat.end(),t.begin(),temp.begin(),thrust::multiplies<Scalar>());
		// rho1 = -w * thrust::reduce(thrust::device,temp.begin(),temp.end());
		computeX<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&xx[0]),thrust::raw_pointer_cast(&p[0]),thrust::raw_pointer_cast(&s[0]),xx.size(),alpha,w);
		computeR<<<blockSize,threadSize>>>(thrust::raw_pointer_cast(&r[0]),thrust::raw_pointer_cast(&s[0]),thrust::raw_pointer_cast(&t[0]),r.size(),w);
		iter++;
		thrust::transform(thrust::device,r.begin(),r.end(),r.begin(),temp.begin(),thrust::multiplies<Scalar>());
		norm = thrust::reduce(thrust::device,temp.begin(),temp.end());
		norm = std::sqrt(norm);
	}
	x.setvalues({xx.begin(),xx.end()});
}



// inline void processCUDAError(cusparseStatus_t err, const char *file, int line) {
//     if (err == CUSPARSE_STATUS_SUCCESS)
//         return;
//     std::cerr << "CUDA error: " << cusparseGetErrorString(err) << " happened at line: " << line
//               << " in file: " << file << std::endl;
// }
// #define CUSPARSECheck(f) processCUDAError(f, __FILE__, __LINE__)

// void BiCGSTAB(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
// {
// 	printf("BiCGSTAB...\n");
// 	int num_rows = A.m_Mat.size(), num_not_zero = 0;
// 	int m = num_rows;
// 	int num_offsets = m + 1;
	
// 	// CSR Format in Thrust
// 	thrust::device_vector<idxType> A_rows;
// 	A_rows.resize(num_offsets);
// 	thrust::device_vector<idxType> A_columns;
// 	thrust::device_vector<Scalar> A_values, M_values;
// 	// from ellpack to csr
// 	for (int i = 0; i < A.m_Mat.size(); i++) {
// 		for (int j = 0; j < A.m_Mat[i].size(); j++) {
// 			if (A.m_Mat[i][j].second != 0) {
// 				A_columns.push_back(A.m_Mat[i][j].first);
// 				A_values.push_back(A.m_Mat[i][j].second);
// 				num_not_zero++;
// 			}
// 		}
// 		A_rows[i + 1] = num_not_zero;
// 	}
// 	M_values.resize(A_values.size());
// 	thrust::copy(A_values.begin(), A_values.end(), M_values.begin());

// 	CudaVector B(b.size()), R(b.size()), R0(b.size()), X(b.size()), V(b.size()), P(b.size()), P_aux(b.size()), S(b.size()), S_aux(b.size()), T(b.size()), temp(b.size());
// 	// std::vector<Scalar> b_vec = b.generateScalar();
// 	// thrust::copy(b_vec.begin(), b_vec.end(), B.values.begin());
// 	// thrust::copy(b_vec.begin(), b_vec.end(), R0.values.begin());

// 	cublasHandle_t cublasHandle = NULL;
// 	cusparseHandle_t cusparseHandle = NULL;
// 	cublasCreate(&cublasHandle);
// 	cusparseCreate(&cusparseHandle);
	
// 	cusparseCreateDnVec(&B.vec, m, thrust::raw_pointer_cast(&B.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&R.vec, m, thrust::raw_pointer_cast(&R.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&X.vec, m, thrust::raw_pointer_cast(&X.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&R0.vec, m, thrust::raw_pointer_cast(&R0.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&V.vec, m, thrust::raw_pointer_cast(&V.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&P.vec, m, thrust::raw_pointer_cast(&P.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&P_aux.vec, m, thrust::raw_pointer_cast(&P_aux.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&S.vec, m, thrust::raw_pointer_cast(&S.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&S_aux.vec, m, thrust::raw_pointer_cast(&S_aux.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&T.vec, m, thrust::raw_pointer_cast(&T.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&temp.vec, m, thrust::raw_pointer_cast(&temp.values[0]), CUDA_R_64F);

// 	std::vector<Scalar> b_vec = b.generateScalar();
// 	thrust::copy(b_vec.begin(), b_vec.end(), B.values.begin());

// 	cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
//     // IMPORTANT: Upper/Lower triangular decompositions of A
//     //            (matM_lower, matM_upper) must use two distinct descriptors
//     cusparseSpMatDescr_t matA, matM_lower, matM_upper;
//     cusparseMatDescr_t   matLU;
//     thrust::device_vector<Scalar> M_rows=A_rows, M_columns=A_columns;
//     cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
//     cusparseDiagType_t   diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
//     cusparseFillMode_t   fill_upper    = CUSPARSE_FILL_MODE_UPPER;
//     cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

// 	// A
// 	cusparseCreateCsr(&matA, m, m, num_not_zero, thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), thrust::raw_pointer_cast(&A_values[0]), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, baseIdx, CUDA_R_64F);
// 	// M_lower
// 	cusparseCreateCsr(&matM_lower, m, m, num_not_zero, thrust::raw_pointer_cast(&M_rows[0]), thrust::raw_pointer_cast(&M_columns[0]), thrust::raw_pointer_cast(&M_values[0]), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, baseIdx, CUDA_R_64F);
// 	cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
// 	cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));
// 	// M_upper
// 	cusparseCreateCsr(&matM_upper, m, m, num_not_zero, thrust::raw_pointer_cast(&M_rows[0]), thrust::raw_pointer_cast(&M_columns[0]), thrust::raw_pointer_cast(&M_values[0]), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, baseIdx, CUDA_R_64F);
// 	cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper));
// 	cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit));
	
// 	//-------------------------------------------------------------------------
// 	// ### Preparation ### b = A * X
// 	const double Alpha = 0.75;
// 	size_t bufferSizeMV;
// 	double beta = 0.0;
// 	cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, matA, X.vec, &beta, B.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV);
// 	thrust::device_vector<Scalar> bufferMV(bufferSizeMV);
// 	cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, matA, X.vec, &beta, B.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
// 	// X0 = 0
// 	// Perform Incomplete-LU factorization of A (csrilu0) -> M_lower, M_upper
// 	csrilu02Info_t infoM = NULL;
//     int bufferSizeLU = 0;
    
// 	cusparseCreateMatDescr(&matLU);
//     cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL);
//     cusparseSetMatIndexBase(matLU, baseIdx);
//     cusparseCreateCsrilu02Info(&infoM);

// 	cusparseDcsrilu02_bufferSize(cusparseHandle, m, num_not_zero, matLU, thrust::raw_pointer_cast(&M_values[0]), thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), infoM, &bufferSizeLU);
// 	thrust::device_vector<Scalar> bufferLU(bufferSizeLU);
// 	cusparseDcsrilu02_analysis(cusparseHandle, m, num_not_zero, matLU, thrust::raw_pointer_cast(&M_values[0]), thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL, thrust::raw_pointer_cast(&bufferLU[0]));

// 	int structural_zero;
// 	cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM, &structural_zero);

// 	// M = L * U
// 	cusparseDcsrilu02(cusparseHandle, m, num_not_zero, matLU, thrust::raw_pointer_cast(&M_values[0]), thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL, thrust::raw_pointer_cast(&bufferLU[0]));

// 	// find numerical zero;
// 	int numerical_zero;
// 	cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM, &numerical_zero);
// 	cusparseDestroyMatDescr(matLU);

// 	//-----------------------------------------------------------------------
// 	// ### BiCGSTAB ###
// 	printf("BiCGSTAB loop:\n");
// 	// gpu_BiCGSTAB(cublasHandle, cusparseHandle, m, matA, matM_lower, matM_upper, B, X, R0, R, P, P_aux, S, S_aux, V, T, temp, bufferMV, limit, tolerance, iter, norm);

// 	const double zero      = 0.0;
//     const double one       = 1.0;
//     const double minus_one = -1.0;
//     //--------------------------------------------------------------------------
//     // Create opaque data structures that holds analysis data between calls
//     double              coeff_tmp;
//     size_t              bufferSizeL, bufferSizeU;
//     // void*               bufferL, *bufferU;
//     cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
//     cusparseSpSV_createDescr(&spsvDescrL);
//     cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_lower, P.vec, temp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL);
// 	thrust::device_vector<Scalar> bufferL(bufferSizeL);
// 	thrust::device_vector<Scalar> bufferU(bufferSizeL);
// 	thrust::copy(B.values.begin(), B.values.end(), R0.values.begin());
// 	cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_lower, P.vec, temp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, thrust::raw_pointer_cast(&bufferL[0]));

// 	// Calculate UPPER buffersize
// 	cusparseSpSV_createDescr(&spsvDescrU);
// 	cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_upper, temp.vec, P_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU);
	
// 	// auto error = cudaGetLastError();
// 	cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &coeff_tmp, matM_upper, temp.vec, P_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, thrust::raw_pointer_cast(&bufferU[0]));
// 	//--------------------------------------------------------------------------
// 	// ### 1 ### R0 = b - A * X0 (using initial guess in X)
// 	//    (a) copy b in R0
// 	//    (b) compute R = -A * X0 + R
// 	cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, X.vec, &one, R0.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));

// 	//--------------------------------------------------------------------------
// 	double alpha, delta, delta_prev, omega;
// 	cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&R0.values[0]), 1, thrust::raw_pointer_cast(&R0.values[0]), 1, &delta);
// 	delta_prev = delta;
// 	// R = R0
// 	thrust::copy(R0.values.begin(), R0.values.end(), R.values.begin());
// 	//--------------------------------------------------------------------------
// 	// nrm_R0 = ||R||
// 	double nrm_R;
// 	cublasDnrm2(cublasHandle, m, thrust::raw_pointer_cast(&R0.values[0]), 1, &nrm_R);
// 	double threshold = tolerance * nrm_R;
// 	printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
// 	//--------------------------------------------------------------------------
// 	// ### 2 ### repeat until convergence based on max iterations and
// 	//           and relative residual
// 	for (int i = 1; i <= limit; i++) {
// 		// printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
// 		//----------------------------------------------------------------------
// 		// ### 4, 7 ### P_i = R_i
// 		thrust::copy(R.values.begin(), R.values.end(), P.values.begin());
// 		if (i > 1) {
// 			//------------------------------------------------------------------
// 			// ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
// 			//    (a) delta_i = (R'_0, R_i-1)
// 			cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&R0.values[0]), 1, thrust::raw_pointer_cast(&R.values[0]), 1, &delta);
// 			//    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
// 			double beta = (delta / delta_prev) * (alpha / omega);
// 			delta_prev  = delta;
// 			//------------------------------------------------------------------
// 			// ### 7 ### P = R + beta * (P - omega * V)
// 			//    (a) P = - omega * V + P
// 			double minus_omega = -omega;
// 			cublasDaxpy(cublasHandle, m, &minus_omega, thrust::raw_pointer_cast(&V.values[0]), 1, thrust::raw_pointer_cast(&P.values[0]), 1);
// 			//    (b) P = beta * P
// 			cublasDscal(cublasHandle, m, &beta, thrust::raw_pointer_cast(&P.values[0]), 1);
// 			//    (c) P = R + P
// 			cublasDaxpy(cublasHandle, m, &one, thrust::raw_pointer_cast(&R.values[0]), 1, thrust::raw_pointer_cast(&P.values[0]), 1);
// 		}
// 		//----------------------------------------------------------------------
// 		// ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
// 		//    (a) M_L^-1 P_i => tmp    (triangular solver)
// 		// thrust::fill(temp.values.begin(), temp.values.end(), 0);
// 		// thrust::fill(P_aux.values.begin(), P_aux.values.end(), 0);
// 		cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lower, P.vec, temp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);
// 		//    (b) M_U^-1 tmp => P_aux    (triangular solver)
// 		cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_upper, temp.vec, P_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU);
// 		//----------------------------------------------------------------------
// 		// ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
// 		//    (a) V = A * P_aux
// 		cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, P_aux.vec, &zero, V.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
// 		//    (b) denominator = R'0 * V
// 		double denominator;
// 		cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&R0.values[0]), 1, thrust::raw_pointer_cast(&V.values[0]), 1, &denominator);
// 		alpha = delta / denominator;
// 		// PRINT_INFO(delta)
// 		// PRINT_INFO(alpha)
// 		//----------------------------------------------------------------------
// 		// ### 11 ###  X_i = X_i-1 + alpha * P_aux
// 		cublasDaxpy(cublasHandle, m, &alpha, thrust::raw_pointer_cast(&P_aux.values[0]), 1, thrust::raw_pointer_cast(&X.values[0]), 1);
// 		//----------------------------------------------------------------------
// 		// ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
// 		//    (a) S = R_i-1
// 		thrust::copy(R.values.begin(), R.values.end(), S.values.begin());
// 		//    (b) S = -alpha * V + R_i-1
// 		double minus_alpha = -alpha;
// 		cublasDaxpy(cublasHandle, m, &minus_alpha, thrust::raw_pointer_cast(&V.values[0]), 1, thrust::raw_pointer_cast(&S.values[0]), 1);
// 		//----------------------------------------------------------------------
// 		// ### 13 ###  check ||S|| < threshold
// 		double nrm_S;
// 		cublasDnrm2(cublasHandle, m, thrust::raw_pointer_cast(&S.values[0]), 1, &nrm_S);
// 		// PRINT_INFO(nrm_S)
// 		iter++;
// 		if (nrm_S < tolerance) {
// 			break;
// 		}
// 		//----------------------------------------------------------------------
// 		// ### 14 ### S_aux = M_U^-1 M_L^-1 S
// 		//    (a) M_L^-1 S => tmp    (triangular solver)
// 		// cudaMemset(tmp.ptr, 0x0, m * sizeof(double));
// 		// cudaMemset(S_aux.ptr, 0x0, m * sizeof(double));
// 		cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_lower, S.vec, temp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);
// 		//    (b) M_U^-1 tmp => S_aux    (triangular solver)
// 		cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matM_upper, temp.vec, S_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU);
// 		//----------------------------------------------------------------------
// 		// ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
// 		//    (a) T = A * S_aux
// 		cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, S_aux.vec, &zero, T.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
// 		//    (b) omega_num = (A * S_aux, s)
// 		double omega_num, omega_den;
// 		cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&T.values[0]), 1, thrust::raw_pointer_cast(&S.values[0]), 1, &omega_num);
// 		//    (c) omega_den = (A * S_aux, A * S_aux)
// 		cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&T.values[0]), 1, thrust::raw_pointer_cast(&T.values[0]), 1, &omega_den);
// 		//    (d) omega = omega_num / omega_den
// 		omega = omega_num / omega_den;
// 		// PRINT_INFO(omega)
// 		// ---------------------------------------------------------------------
// 		// ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
// 		//    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
// 		//        X_i = omega * S_aux + X_i
// 		cublasDaxpy(cublasHandle, m, &omega, thrust::raw_pointer_cast(&S_aux.values[0]), 1, thrust::raw_pointer_cast(&X.values[0]), 1);
// 		// ---------------------------------------------------------------------
// 		// ### 17 ###  R_i+1 = S - omega * (A * S_aux)
// 		//    (a) copy S in R
// 		thrust::copy(S.values.begin(), S.values.end(), R.values.begin());
// 		//    (a) R_i+1 = -omega * T + R
// 		double minus_omega = -omega;
// 		cublasDaxpy(cublasHandle, m, &minus_omega, thrust::raw_pointer_cast(&T.values[0]), 1, thrust::raw_pointer_cast(&R.values[0]), 1);
// 		// ---------------------------------------------------------------------
// 		// ### 18 ###  check ||R_i|| < threshold
// 		cublasDnrm2(cublasHandle, m, thrust::raw_pointer_cast(&R.values[0]), 1, &nrm_R);
// 		// PRINT_INFO(nrm_R)
// 		if (nrm_R < tolerance) {
// 			break;
// 		}
// 	}
// 	//--------------------------------------------------------------------------
// 	printf("Check Solution\n"); // ||R = b - A * X||
// 	//    (a) copy b in R
// 	thrust::copy(B.values.begin(), B.values.end(), R.values.begin());
// 	// R = -A * X + R
// 	cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, X.vec, &one, R.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
// 	// check ||R||
// 	cublasDnrm2(cublasHandle, m, thrust::raw_pointer_cast(&R.values[0]), 1, &nrm_R);
// 	norm = nrm_R;
// 	printf("Final iterations = %d\nFinal error norm = %e\n", iter, nrm_R);
// 	//--------------------------------------------------------------------------
// 	 cusparseSpSV_destroyDescr(spsvDescrL);
// 	 cusparseSpSV_destroyDescr(spsvDescrU);
	
	
// 	x.setvalues({X.values.begin(),X.values.end()});

// 	cusparseDestroyDnVec(B.vec);
//     cusparseDestroyDnVec(X.vec);
//     cusparseDestroyDnVec(R.vec);
//     cusparseDestroyDnVec(R0.vec);
//     cusparseDestroyDnVec(P.vec);
//     cusparseDestroyDnVec(P_aux.vec);
//     cusparseDestroyDnVec(S.vec);
//     cusparseDestroyDnVec(S_aux.vec);
//     cusparseDestroyDnVec(V.vec);
//     cusparseDestroyDnVec(T.vec);
//     cusparseDestroyDnVec(temp.vec);
//     cusparseDestroySpMat(matA);
//     cusparseDestroySpMat(matM_lower);
//     cusparseDestroySpMat(matM_upper);
//     cusparseDestroy(cusparseHandle);
//     cublasDestroy(cublasHandle);

// 	return ;

// }

// void PCG_ICC(const SymetrixSparseMatrix& A,Vector& x,const Vector& b,double tolerance,int limit,int& iter,double& norm)
// {	
// 	printf("PCG_ICC...\n");
// 	int num_rows = A.m_Mat.size(), num_not_zero = 0;
// 	int m = num_rows;
// 	int num_offsets = m + 1;
	
// 	// CSR Format in Thrust
// 	thrust::device_vector<idxType> A_rows;
// 	A_rows.resize(num_offsets);
// 	thrust::device_vector<idxType> A_columns;
// 	thrust::device_vector<Scalar> A_values, L_values;
// 	// from ellpack to csr
// 	for (int i = 0; i < A.m_Mat.size(); i++) {
// 		for (int j = 0; j < A.m_Mat[i].size(); j++) {
// 			if (A.m_Mat[i][j].second != 0) {
// 				A_columns.push_back(A.m_Mat[i][j].first);
// 				A_values.push_back(A.m_Mat[i][j].second);
// 				num_not_zero++;
// 			}
// 		}
// 		A_rows[i + 1] = num_not_zero;
// 	}
// 	L_values.resize(A_values.size());
// 	thrust::copy(A_values.begin(), A_values.end(), L_values.begin());

// 	CudaVector B(b.size()), R(b.size()), R_aux(b.size()), X(b.size()), P(b.size()), T(b.size()), tmp(b.size());
	
// 	//--------------------------------------------------------------------------
//     // ### cuSPARSE Handle and descriptors initialization ###
//     // create the test matrix on the host
//     cublasHandle_t   cublasHandle   = NULL;
//     cusparseHandle_t cusparseHandle = NULL;
//     cublasCreate(&cublasHandle);
//     cusparseCreate(&cusparseHandle);
//     // Create dense vectors
// 	cusparseCreateDnVec(&B.vec, m, thrust::raw_pointer_cast(&B.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&X.vec, m, thrust::raw_pointer_cast(&X.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&R.vec, m, thrust::raw_pointer_cast(&R.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&R_aux.vec, m, thrust::raw_pointer_cast(&R_aux.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&P.vec, m, thrust::raw_pointer_cast(&P.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&T.vec, m, thrust::raw_pointer_cast(&T.values[0]), CUDA_R_64F);
// 	cusparseCreateDnVec(&tmp.vec, m, thrust::raw_pointer_cast(&tmp.values[0]), CUDA_R_64F);

// 	std::vector<Scalar> b_vec = b.generateScalar();
// 	thrust::copy(b_vec.begin(), b_vec.end(), B.values.begin());

//     cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
//     cusparseSpMatDescr_t matA, matL;
//     thrust::device_vector<Scalar> L_rows = A_rows, L_columns = A_columns;
//     cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
//     cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
//     // A
//     cusparseCreateCsr(&matA, m, m, num_not_zero, thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), thrust::raw_pointer_cast(&A_values[0]), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, baseIdx, CUDA_R_64F);
//     // L
//     cusparseCreateCsr(&matL, m, m, num_not_zero, thrust::raw_pointer_cast(&L_rows[0]), thrust::raw_pointer_cast(&L_columns[0]), thrust::raw_pointer_cast(&L_values[0]), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, baseIdx, CUDA_R_64F);
//     cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
//     cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit));
//     //--------------------------------------------------------------------------
//     // ### Preparation ### b = A * X
//     const double alpha = 0.75;
//     size_t       bufferSizeMV;
//     double       beta = 0.0;
//     cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, X.vec, &beta, B.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV);
//     thrust::device_vector<Scalar> bufferMV(bufferSizeMV);
//     cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, X.vec, &beta, B.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
//     // X0 = 0
// 	thrust::fill(X.values.begin(), X.values.end(), 0.0);
//     //--------------------------------------------------------------------------
//     // Perform Incomplete-Cholesky factorization of A (csric0) -> L, L^T
//     cusparseMatDescr_t descrM;
//     csric02Info_t      infoM        = NULL;
//     int                bufferSizeIC = 0;
//     cusparseCreateMatDescr(&descrM);
//     cusparseSetMatIndexBase(descrM, baseIdx);
//     cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
//     cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER);
//     cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT);
//     cusparseCreateCsric02Info(&infoM);

//     cusparseDcsric02_bufferSize(cusparseHandle, m, num_not_zero, descrM, thrust::raw_pointer_cast(&L_values[0]), thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), infoM, &bufferSizeIC);
 
//     thrust::device_vector<Scalar> bufferIC(bufferSizeIC);
// 	cusparseDcsric02_analysis(cusparseHandle, m, num_not_zero, descrM, thrust::raw_pointer_cast(&L_values[0]), thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, thrust::raw_pointer_cast(&bufferIC[0]));
//     int structural_zero;
//     cusparseXcsric02_zeroPivot(cusparseHandle, infoM, &structural_zero);
//     // M = L * L^T
//     cusparseDcsric02(cusparseHandle, m, num_not_zero, descrM, thrust::raw_pointer_cast(&L_values[0]), thrust::raw_pointer_cast(&A_rows[0]), thrust::raw_pointer_cast(&A_columns[0]), infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, thrust::raw_pointer_cast(&bufferIC[0]));
//     // Find numerical zero
//     int numerical_zero;
//     cusparseXcsric02_zeroPivot(cusparseHandle, infoM, &numerical_zero);

//     cusparseDestroyCsric02Info(infoM);
//     cusparseDestroyMatDescr(descrM);
//     //--------------------------------------------------------------------------
//     // ### Run CG computation ###
//     printf("PCG loop:\n");
//     // gpu_CG(cublasHandle, cusparseHandle, m,
//     //        matA, matL, B, X, R, R_aux, P, T,
//     //        tmp, bufferMV, maxIterations, tolerance);

// 	const double zero      = 0.0;
//     const double one       = 1.0;
//     const double minus_one = -1.0;
//     //--------------------------------------------------------------------------
//     // ### 1 ### R0 = b - A * X0 (using initial guess in X)
//     //    (a) copy b in R0
//     thrust::copy(B.values.begin(), B.values.end(), R.values.begin());

//     //    (b) compute R = -A * X0 + R
//     cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, X.vec, &one, R.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
//     //--------------------------------------------------------------------------
//     // ### 2 ### R_i_aux = L^-1 L^-T R_i
//     size_t              bufferSizeL, bufferSizeLT;
//     cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
//     //    (a) L^-T tmp => R_i_aux    (triangular solver)
//     cusparseSpSV_createDescr(&spsvDescrLT);
//     cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, R.vec, tmp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT);
//     thrust::device_vector<Scalar> bufferLT(bufferSizeLT);
//     thrust::fill(tmp.values.begin(), tmp.values.end(), 0.0);
//     CUSPARSECheck( cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, R.vec, tmp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, thrust::raw_pointer_cast(&bufferLT[0])) );
// 	cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, R.vec, tmp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT);

//     //    (b) L^-T R_i => tmp    (triangular solver)
//     cusparseSpSV_createDescr(&spsvDescrL);
//     cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, tmp.vec, R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL);
//     thrust::device_vector<Scalar> bufferL(bufferSizeL);
//     cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, tmp.vec, R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, thrust::raw_pointer_cast(&bufferL[0]));
//     thrust::fill(R_aux.values.begin(), R_aux.values.end(), 0.0);
//     cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, tmp.vec, R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);
//     //--------------------------------------------------------------------------
//     // ### 3 ### P0 = R0_aux
// 	thrust::copy(R_aux.values.begin(), R_aux.values.end(), P.values.begin());
//     //--------------------------------------------------------------------------
//     // nrm_R0 = ||R||
//     double nrm_R;
//     cublasDnrm2(cublasHandle, m, thrust::raw_pointer_cast(&R.values[0]), 1, &nrm_R);
//     double threshold = tolerance * nrm_R;
//     printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
//     //--------------------------------------------------------------------------
//     double delta;
//     cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&R.values[0]), 1, thrust::raw_pointer_cast(&R.values[0]), 1, &delta);
//     //--------------------------------------------------------------------------
//     // ### 4 ### repeat until convergence based on max iterations and
//     //           and relative residual
//     for (int i = 1; i <= limit; i++) {
//         // printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
//         //----------------------------------------------------------------------
//         // ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
//         //     (a) T  = A * P_i
//         cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, P.vec, &zero, T.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
//         //     (b) denominator = (T, P_i)
//         double denominator;
//         cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&T.values[0]), 1, thrust::raw_pointer_cast(&P.values[0]), 1, &denominator);
//         //     (c) alpha = delta / denominator
//         double alpha = delta / denominator;
//         // PRINT_INFO(delta)
//         // PRINT_INFO(denominator)
//         // PRINT_INFO(alpha)
//         //----------------------------------------------------------------------
//         // ### 6 ###  X_i+1 = X_i + alpha * P
//         //    (a) X_i+1 = -alpha * T + X_i
//         cublasDaxpy(cublasHandle, m, &alpha, thrust::raw_pointer_cast(&P.values[0]), 1, thrust::raw_pointer_cast(&X.values[0]), 1);
//         //----------------------------------------------------------------------
//         // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
//         //    (a) R_i+1 = -alpha * T + R_i
//         double minus_alpha = -alpha;
//         cublasDaxpy(cublasHandle, m, &minus_alpha, thrust::raw_pointer_cast(&T.values[0]), 1, thrust::raw_pointer_cast(&R.values[0]), 1);
//         //----------------------------------------------------------------------
//         // ### 8 ###  check ||R_i+1|| < threshold
//         cublasDnrm2(cublasHandle, m, thrust::raw_pointer_cast(&R.values[0]), 1, &nrm_R);
//         // PRINT_INFO(nrm_R)
// 		iter++;
//         if (nrm_R < threshold)
// 			break;
//         //----------------------------------------------------------------------
//         // ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
//         //    (a) L^-T R_i+1 => tmp    (triangular solver)
// 		thrust::fill(tmp.values.begin(), tmp.values.end(), 0);
// 		thrust::fill(R_aux.values.begin(), R_aux.values.end(), 0);
//         cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL, R.vec, tmp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);
//         //    (b) L^-T tmp => R_aux_i+1    (triangular solver)
//         cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL, tmp.vec, R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT);
//         //----------------------------------------------------------------------
//         // ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
//         //    (a) delta_new => (R_i+1, R_aux_i+1)
//         double delta_new;
//         cublasDdot(cublasHandle, m, thrust::raw_pointer_cast(&R.values[0]), 1, thrust::raw_pointer_cast(&R_aux.values[0]), 1, &delta_new);
//         //    (b) beta => delta_new / delta
//         double beta = delta_new / delta;
//         // PRINT_INFO(delta_new)
//         // PRINT_INFO(beta)
//         delta       = delta_new;
//         //----------------------------------------------------------------------
//         // ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
//         //    (a) copy R_aux_i+1 in P_i
// 		thrust::copy(R_aux.values.begin(), R_aux.values.end(), P.values.begin());
//         //    (b) P_i+1 = beta * P_i + R_aux_i+1
//         cublasDaxpy(cublasHandle, m, &beta, thrust::raw_pointer_cast(&P.values[0]), 1, thrust::raw_pointer_cast(&P.values[0]), 1);
//     }
//     //--------------------------------------------------------------------------
//     printf("Check Solution\n"); // ||R = b - A * X||
//     //    (a) copy b in R
// 	thrust::copy(B.values.begin(), B.values.end(), R.values.begin());
//     // R = -A * X + R
//     cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, X.vec, &one, R.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(&bufferMV[0]));
//     // check ||R||
//     cublasDnrm2(cublasHandle, m, thrust::raw_pointer_cast(&R.values[0]), 1, &nrm_R);
// 	norm = nrm_R;

//     printf("Final iter: %d error norm = %e\n", iter, nrm_R);
//     //--------------------------------------------------------------------------
//     cusparseSpSV_destroyDescr(spsvDescrL);
//     cusparseSpSV_destroyDescr(spsvDescrLT);

// 	x.setvalues({X.values.begin(), X.values.end()});

//     //--------------------------------------------------------------------------
//     // ### Free resources ###
//     cusparseDestroyDnVec(B.vec);
//     cusparseDestroyDnVec(X.vec);
//     cusparseDestroyDnVec(R.vec);
//     cusparseDestroyDnVec(R_aux.vec);
//     cusparseDestroyDnVec(P.vec);
//     cusparseDestroyDnVec(T.vec);
//     cusparseDestroyDnVec(tmp.vec);
//     cusparseDestroySpMat(matA);
//     cusparseDestroySpMat(matL);
//     cusparseDestroy(cusparseHandle);
//     cublasDestroy(cublasHandle);

// 	return ;
// }