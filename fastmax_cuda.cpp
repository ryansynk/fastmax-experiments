#include <iostream>
#include <vector>
#include <math.h>
#include <torch/extension.h>
#include <iostream>
using namespace std;


// torch::Tensor forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor rpe_matrix, torch::Tensor cons, torch::Tensor lin, torch::Tensor quad, bool mask);
torch::Tensor forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor drop_noise, torch::Tensor rpe_matrix, bool mask,float dropout, bool normalize, float temperature);
// vector<torch::Tensor> backward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, torch::Tensor gradq, torch::Tensor gradk, torch::Tensor gradv, torch::Tensor gradq_coeffs0, torch::Tensor gradq_coeffs1, torch::Tensor gradk_coeffs0v, torch::Tensor gradk_coeffs0o, torch::Tensor gradk_coeffs1v, torch::Tensor gradk_coeffs1o, torch::Tensor gradv_coeffs0, torch::Tensor gradv_coeffs1, torch::Tensor gradv_coeffs2, bool mask);
vector<torch::Tensor> backward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, bool mask);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// torch::Tensor forwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor rpe_matrix, torch::Tensor cons, torch::Tensor lin, torch::Tensor quad, bool mask) {
torch::Tensor forwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor drop_noise, torch::Tensor rpe_matrix, bool mask, float dropout, bool normalize, float temperature){

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  // return forward_cuda(q, k, v, o, rpe_matrix, cons, lin, quad, mask);
  return forward_cuda(q, k, v, drop_noise, rpe_matrix,mask,dropout,normalize,temperature);
}

// vector<torch::Tensor> backwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, torch::Tensor gradq, torch::Tensor gradk, torch::Tensor gradv, torch::Tensor gradq_coeffs0, torch::Tensor gradq_coeffs1, torch::Tensor gradk_coeffs0v, torch::Tensor gradk_coeffs0o, torch::Tensor gradk_coeffs1v, torch::Tensor gradk_coeffs1o, torch::Tensor gradv_coeffs0, torch::Tensor gradv_coeffs1, torch::Tensor gradv_coeffs2, bool mask){
vector<torch::Tensor> backwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, bool mask){
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  // return backward_cuda(q, k, v, o, grad_output, gradq, gradk, gradv, gradq_coeffs0, gradq_coeffs1, gradk_coeffs0v, gradk_coeffs0o, gradk_coeffs1v, gradk_coeffs1o, gradv_coeffs0, gradv_coeffs1, gradv_coeffs2, mask);
  return backward_cuda(q, k, v, o, grad_output, mask);
}

PYBIND11_MODULE(fastmax_cuda, m) {
  m.def("forwardpass", &forwardpass, "forwardpass");
  m.def("backwardpass", &backwardpass, "backwardpass");
}
