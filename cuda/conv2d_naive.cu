// CUDA/C++ educational reference: not the historical project's original code.
// One thread computes one output of a 5x5 LeNet-style convolution.
// This is NOT an optimized GPU kernel; it includes a CPU correctness check.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
  std::cerr << cudaGetErrorString(e) << " at line " << __LINE__ << "\n"; \
  std::exit(1); }} while(0)

constexpr int N=1, CI=1, CO=6, H=28, W=28, K=5, P=2;
constexpr int XS=N*CI*H*W, WS=CO*CI*K*K, YS=N*CO*H*W;
__global__ void conv5x5(const float* x, const float* weights, float* y) {
  int ox = blockIdx.x*blockDim.x + threadIdx.x;
  int oy = blockIdx.y*blockDim.y + threadIdx.y;
  int oc = blockIdx.z % CO;
  int n = blockIdx.z / CO;
  if (ox>=W || oy>=H) return;
  float acc=0;
  for(int ic=0; ic<CI; ++ic)
    for(int ky=0; ky<K; ++ky)
      for(int kx=0; kx<K; ++kx) {
        int iy=oy+ky-P, ix=ox+kx-P;
        if(iy>=0 && iy<H && ix>=0 && ix<W)
          acc += x[((n*CI+ic)*H+iy)*W+ix] *
                 weights[((oc*CI+ic)*K+ky)*K+kx];
      }
  y[((n*CO+oc)*H+oy)*W+ox]=acc;
}
int main(){
  std::vector<float> x(XS), w(WS), y(YS), reference(YS);
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> dist(-1.f,1.f);
  for(float& a:x) a=dist(generator);
  for(float& a:w) a=dist(generator);
  for(int oc=0;oc<CO;++oc)
    for(int oy=0;oy<H;++oy)
      for(int ox=0;ox<W;++ox) {
        float acc=0;
        for(int ic=0;ic<CI;++ic)
          for(int ky=0;ky<K;++ky)
            for(int kx=0;kx<K;++kx) {
              int iy=oy+ky-P,ix=ox+kx-P;
              if(iy>=0 && iy<H && ix>=0 && ix<W)
                acc+=x[(ic*H+iy)*W+ix]*w[((oc*CI+ic)*K+ky)*K+kx];
            }
        reference[(oc*H+oy)*W+ox]=acc;
      }
  float *dx=nullptr,*dw=nullptr,*dy=nullptr;
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dx),XS*sizeof(float)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dw),WS*sizeof(float)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dy),YS*sizeof(float)));
  CHECK(cudaMemcpy(dx,x.data(),XS*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dw,w.data(),WS*sizeof(float),cudaMemcpyHostToDevice));
  dim3 block(16,16), grid((W+15)/16,(H+15)/16,N*CO);
  for(int i=0;i<10;++i) conv5x5<<<grid,block>>>(dx,dw,dy);
  CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  for(int i=0;i<100;++i) conv5x5<<<grid,block>>>(dx,dw,dy);
  CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
  float ms=0;
  CHECK(cudaEventElapsedTime(&ms,start,stop));
  CHECK(cudaMemcpy(y.data(),dy,YS*sizeof(float),cudaMemcpyDeviceToHost));
  float max_error=0;
  for(int i=0;i<YS;++i) max_error=std::max(max_error,std::abs(y[i]-reference[i]));
  std::cout << "Average CUDA kernel time (ms): " << ms/100.f << "\n";
  std::cout << "Max absolute error vs CPU: " << max_error << "\n";
  CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
  CHECK(cudaFree(dx)); CHECK(cudaFree(dw)); CHECK(cudaFree(dy));
  return max_error<1.e-4f ? EXIT_SUCCESS : EXIT_FAILURE;
}
