// RWKV7 recurrence adapted from RWKV-Vibe/RWKV-LM-V7 (Apache-2.0).
// Source: https://github.com/RWKV-Vibe/RWKV-LM-V7/blob/
// 665472dab30952de9379a3a3a01eaa3453f1ad4e/cuda/rwkv7_clampw.cu
// See LICENSE-RWKV7. Adaptations: explicit nonzero s0/sT, [value,key] layout,
// direct backward with FP32 history, arbitrary tails, current CUDA stream.
// Keep PyTorch headers in the C++ translation unit for nvcc compatibility.
#include <cuda_runtime.h>
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#include <cuda_bf16.h>
constexpr int MAX_N = 128;

template<typename scalar_t>
__global__ void forward_kernel(int T, int H, int N, const scalar_t* r, const scalar_t* w,
 const scalar_t* k, const scalar_t* v, const scalar_t* a, const scalar_t* b,
 const float* s0, scalar_t* y, float* end, float* hist) {
    int bh=blockIdx.x, batch=bh/H, head=bh%H, i=threadIdx.x;
    __shared__ float sr[MAX_N], sw[MAX_N], sk[MAX_N], sa[MAX_N], sb[MAX_N];
    float state[MAX_N];
    long base=(long)bh*(T+1)*N*N;
    for(int j=0;j<N;j++) {
        state[j]=s0[(long)bh*N*N+i*N+j];
        if(hist) hist[base+i*N+j]=state[j];
    }
    for(int t=0;t<T;t++) {
        long idx=((long)batch*T+t)*H*N+head*N+i;
        __syncthreads();
        sr[i]=(float)r[idx]; sw[i]=expf(-0.6065306597126334f/(1.f+expf(-(float)w[idx])));
        sk[i]=(float)k[idx]; sa[i]=(float)a[idx]; sb[i]=(float)b[idx];
        __syncthreads();
        float dot=0.f, out=0.f, vi=(float)v[idx];
        for(int j=0;j<N;j++) dot+=state[j]*sa[j];
        for(int j=0;j<N;j++) {
            state[j]=state[j]*sw[j]+dot*sb[j]+vi*sk[j];
            out+=state[j]*sr[j];
            if(hist) hist[base+(long)(t+1)*N*N+i*N+j]=state[j];
        }
        y[idx]=(scalar_t)out;
    }
    for(int j=0;j<N;j++) end[(long)bh*N*N+i*N+j]=state[j];
}

template<typename scalar_t>
__global__ void backward_kernel(int T, int H, int N, const scalar_t* r, const scalar_t* w,
 const scalar_t* k, const scalar_t* v, const scalar_t* a, const scalar_t* b,
 const float* hist, const scalar_t* dy, const float* dend,
 float* dr,float* dw,float* dk,float* dv,float* da,float* db,float* ds0) {
    int bh=blockIdx.x,batch=bh/H,head=bh%H,i=threadIdx.x;
    __shared__ float sr[MAX_N],sw[MAX_N],sig[MAX_N],sk[MAX_N],sa[MAX_N],sb[MAX_N];
    float ds[MAX_N];
    for(int j=0;j<N;j++) ds[j]=dend[(long)bh*N*N+i*N+j];
    long base=(long)bh*(T+1)*N*N;
    for(int t=T-1;t>=0;t--) {
        long idx=((long)batch*T+t)*H*N+head*N+i;
        long vec=((long)batch*T+t)*H*N+head*N;
        __syncthreads();
        sr[i]=(float)r[idx];sig[i]=1.f/(1.f+expf(-(float)w[idx]));
        sw[i]=expf(-0.6065306597126334f*sig[i]);sk[i]=(float)k[idx];
        sa[i]=(float)a[idx];sb[i]=(float)b[idx];
        __syncthreads();
        float dyi=(float)dy[idx],vi=(float)v[idx],dot=0.f,dotgrad=0.f,dvi=0.f;
        long prev=base+(long)t*N*N+i*N, next=prev+N*N;
        for(int j=0;j<N;j++) {
            dot+=hist[prev+j]*sa[j];
            ds[j]+=dyi*sr[j];
            dotgrad+=ds[j]*sb[j];
        }
        for(int j=0;j<N;j++) {
            atomicAdd(dr+vec+j,dyi*hist[next+j]);
            atomicAdd(dw+vec+j,ds[j]*hist[prev+j]*sw[j]*(-0.6065306597126334f)*sig[j]*(1.f-sig[j]));
            atomicAdd(dk+vec+j,ds[j]*vi);
            atomicAdd(db+vec+j,ds[j]*dot);
            atomicAdd(da+vec+j,dotgrad*hist[prev+j]);
            dvi+=ds[j]*sk[j];
            ds[j]=ds[j]*sw[j]+dotgrad*sa[j];
        }
        dv[idx]=dvi;
    }
    for(int j=0;j<N;j++) ds0[(long)bh*N*N+i*N+j]=ds[j];
}

void launch_forward(int B,int T,int H,int N,const void* r,const void* w,const void* k,const void* v,const void* a,const void* b,
 const float* s0,void* y,float* end,float* hist,bool bf16,cudaStream_t stream) {
#define FWD(TYPE) forward_kernel<TYPE><<<B*H,N,0,stream>>>(T,H,N,(const TYPE*)r,(const TYPE*)w,(const TYPE*)k,(const TYPE*)v,(const TYPE*)a,(const TYPE*)b,s0,(TYPE*)y,end,hist)
    if(bf16) { FWD(__nv_bfloat16); } else { FWD(float); }
#undef FWD
}
void launch_backward(int B,int T,int H,int N,const void* r,const void* w,const void* k,const void* v,const void* a,const void* b,
 const float* hist,const void* dy,const float* dend,float* dr,float* dw,float* dk,float* dv,float* da,float* db,float* ds0,bool bf16,cudaStream_t stream) {
#define BWD(TYPE) backward_kernel<TYPE><<<B*H,N,0,stream>>>(T,H,N,(const TYPE*)r,(const TYPE*)w,(const TYPE*)k,(const TYPE*)v,(const TYPE*)a,(const TYPE*)b,hist,(const TYPE*)dy,dend,dr,dw,dk,dv,da,db,ds0)
    if(bf16) { BWD(__nv_bfloat16); } else { BWD(float); }
#undef BWD
}
