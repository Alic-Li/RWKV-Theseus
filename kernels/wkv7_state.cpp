#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

void launch_forward(int,int,int,int,const void*,const void*,const void*,const void*,const void*,const void*,const float*,void*,float*,float*,bool,cudaStream_t);
void launch_backward(int,int,int,int,const void*,const void*,const void*,const void*,const void*,const void*,const float*,const void*,const float*,float*,float*,float*,float*,float*,float*,float*,bool,cudaStream_t);

void check(const torch::Tensor& r, const std::vector<torch::Tensor>& xs) {
    TORCH_CHECK(r.is_cuda() && r.is_contiguous() && r.dim()==4,"expected contiguous CUDA [B,T,H,N]");
    TORCH_CHECK(r.size(1)>0 && r.size(3)>=2 && r.size(3)<=128,"invalid T or head size (2..128)");
    TORCH_CHECK(r.scalar_type()==torch::kBFloat16 || r.scalar_type()==torch::kFloat32,"expected BF16 or FP32");
    for(auto& x:xs) TORCH_CHECK(x.device()==r.device() && x.sizes()==r.sizes() && x.scalar_type()==r.scalar_type() && x.is_contiguous(),"input mismatch");
}
std::vector<torch::Tensor> wkv_forward(torch::Tensor r,torch::Tensor w,torch::Tensor k,torch::Tensor v,torch::Tensor a,torch::Tensor b,torch::Tensor s0,bool save_history) {
    check(r,{w,k,v,a,b}); c10::cuda::CUDAGuard guard(r.device());
    int B=r.size(0),T=r.size(1),H=r.size(2),N=r.size(3);
    TORCH_CHECK(s0.device()==r.device() && s0.scalar_type()==torch::kFloat32 && s0.is_contiguous(),"s0 must be contiguous CUDA FP32");
    TORCH_CHECK(s0.sizes()==torch::IntArrayRef({B,H,N,N}),"invalid s0 shape");
    auto y=torch::empty_like(r), end=torch::empty_like(s0);
    auto hist=save_history ? torch::empty({B,H,T+1,N,N},s0.options()) : torch::empty({0},s0.options());
    launch_forward(B,T,H,N,r.data_ptr(),w.data_ptr(),k.data_ptr(),v.data_ptr(),a.data_ptr(),b.data_ptr(),s0.data_ptr<float>(),y.data_ptr(),end.data_ptr<float>(),save_history ? hist.data_ptr<float>() : nullptr,r.scalar_type()==torch::kBFloat16,at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {y,end,hist};
}
std::vector<torch::Tensor> wkv_backward(torch::Tensor r,torch::Tensor w,torch::Tensor k,torch::Tensor v,torch::Tensor a,torch::Tensor b,torch::Tensor hist,torch::Tensor dy,torch::Tensor dend) {
    check(r,{w,k,v,a,b,dy}); c10::cuda::CUDAGuard guard(r.device());
    int B=r.size(0),T=r.size(1),H=r.size(2),N=r.size(3);
    TORCH_CHECK(hist.device()==r.device() && hist.scalar_type()==torch::kFloat32 && hist.is_contiguous() && hist.sizes()==torch::IntArrayRef({B,H,T+1,N,N}),"invalid history");
    TORCH_CHECK(dend.device()==r.device() && dend.scalar_type()==torch::kFloat32 && dend.is_contiguous() && dend.sizes()==torch::IntArrayRef({B,H,N,N}),"invalid dend");
    auto opts=r.options().dtype(torch::kFloat32);
    std::vector<torch::Tensor> g;
    for(int i=0;i<6;i++) g.push_back(torch::zeros(r.sizes(),opts));
    g.push_back(torch::empty({B,H,N,N},opts));
    launch_backward(B,T,H,N,r.data_ptr(),w.data_ptr(),k.data_ptr(),v.data_ptr(),a.data_ptr(),b.data_ptr(),hist.data_ptr<float>(),dy.data_ptr(),dend.data_ptr<float>(),g[0].data_ptr<float>(),g[1].data_ptr<float>(),g[2].data_ptr<float>(),g[3].data_ptr<float>(),g[4].data_ptr<float>(),g[5].data_ptr<float>(),g[6].data_ptr<float>(),r.scalar_type()==torch::kBFloat16,at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return g;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &wkv_forward);
    m.def("backward", &wkv_backward);
}
