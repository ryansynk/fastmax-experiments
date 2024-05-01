#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.166666;
// __device__ float a2 = 0.145833;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 2;

// // kernel = a0 + a1x + a2x^2
__device__ float a0 = 1.0;
__device__ float a1 = 1.0;
__device__ float a2 = 0.5;
// -lim^2 <= q.k <= lim^2
__device__ float lim = 1;

namespace {
__global__
void calc_unmasked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(outer < d && i < bh){

    // UNMASKED PART ////////////////////////////
    // calc cons denum
    for(int l = 0; l < nq; ++l){
      o[l][i][d] = a0*(nk);
    }

    // calc lin denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk; ++l){
      s[d+outer] += a1*k[l][i][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[outer] = q[l][i][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[l][i][d] += t;
      }
    }

    // calc quad denum
    for(int rr = 0; rr < d/sz; ++rr){
      for(int r = 0; r < sz; ++r) tr[r]= 0;
      for(int l = 0; l < nk;  ++l){
        s[outer] = k[l][i][outer];
        __syncthreads();
        loc1 = rr*sz;
        for(int r = 0; r < sz; ++r){
          tr[r] += s[outer]*s[loc1+r];
        }
      }
      for(int l = 0; l < nq;  ++l){
        s[d+outer] = 0;
        s[outer] = q[l][i][outer];
        __syncthreads();
        loc2 = rr*sz;
        for(int r = 0; r < sz; ++r){
          s[d+outer] += tr[r]*s[outer]*s[loc2+r];
        }
        o[l][i][outer] += s[d+outer];
      }
      __syncthreads();
      for(int l = 0; l < nq; ++l){
        t = 0;
        s[outer] = o[l][i][outer];
        __syncthreads();
        if(outer == 0){
          for(int r = 0; r < d; ++r) t += s[r];
          o[l][i][d] += a2*t;
        }
      }
      __syncthreads();
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk;  ++l){
      t += v[l][i][outer];
    }
    for(int l = 0; l < nq;  ++l){
      o[l][i][outer] = a0*t;
    }

    // calc lin
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = 0; l < nk;  ++l){
        t += k[l][i][m]*v[l][i][outer];
      }
      for(int l = 0; l < nq;  ++l){
        o[l][i][outer] += a1*t*q[l][i][m];
      }
    }

    // calc quad
    for(int m = 0; m < d; ++m){
      for(int rr = 0; rr < d/sz; ++rr){
        for(int r = 0; r < sz; ++r) tr[r]= 0;
        for(int l = 0; l < nk;  ++l){
          s[d+outer] = k[l][i][m]*k[l][i][outer];
          tv = v[l][i][outer];
          __syncthreads();
          loc1 = d+rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[loc1+r]*tv;
          }      
        }
        for(int l = 0; l < nq;  ++l){
          s[outer] = q[l][i][m]*q[l][i][outer];
          __syncthreads();
          t = 0;
          loc2 = rr*sz;
          for(int r = 0; r < sz; ++r){
            t += tr[r]*s[loc2+r];
          }      
          o[l][i][outer] += a2*t;
        }
      }
    }

    for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void calc_masked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(outer < d && i < bh){

    // MASKED PART ////////////////////////////
    // calc cons denum
    for(int l = 0; l < nq; ++l){
      o[l][i][d] = a0*(nk-nq+l+1);
    }

    // calc lin denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk-nq; ++l){
      s[d+outer] += a1*k[l][i][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[d+outer] += a1*k[nk-nq+l][i][outer];
      s[outer] = q[l][i][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[l][i][d] += t;
      }
    }

    // calc quad denum
    for(int rr = 0; rr < d/sz; ++rr){
      for(int r = 0; r < sz; ++r) tr[r]= 0;
      for(int l = 0; l < nk-nq;  ++l){
        s[outer] = k[l][i][outer];
        __syncthreads();
        loc1 = rr*sz;
        for(int r = 0; r < sz; ++r){
          tr[r] += s[outer]*s[loc1+r];
        }
      }
      __syncthreads();
      for(int l = 0; l < nq; ++l){
        s[outer] = k[nk-nq+l][i][outer];
        __syncthreads();
        loc1 = rr*sz;
        for(int r = 0; r < sz; ++r){
          tr[r] += s[outer]*s[loc1+r];
        }
        s[d+outer] = 0;
        s[outer] = q[l][i][outer];
        __syncthreads();
        loc2 = rr*sz;
        for(int r = 0; r < sz; ++r){
          s[d+outer] += tr[r]*s[outer]*s[loc2+r];
        }
        o[l][i][outer] += s[d+outer];
      }
      __syncthreads();
      for(int l = 0; l < nq; ++l){
        t = 0;
        s[outer] = o[l][i][outer];
        __syncthreads();
        if(outer == 0){
          for(int r = 0; r < d; ++r) t += s[r];
          o[l][i][d] += a2*t;
        }
      }

    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk-nq;  ++l){
      t += v[l][i][outer];
    }
    for(int l = 0; l < nq;  ++l){
      t += v[nk-nq+l][i][outer];
      o[l][i][outer] = a0*t;
    }

    // calc lin
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = 0; l < nk-nq;  ++l){
        t += k[l][i][m]*v[l][i][outer];
      }
      for(int l = 0; l < nq;  ++l){
        t += k[nk-nq+l][i][m]*v[nk-nq+l][i][outer];
        o[l][i][outer] += a1*t*q[l][i][m];
      }
    }

    // calc quad
    for(int m = 0; m < d; ++m){
      for(int rr = 0; rr < d/sz; ++rr){
        for(int r = 0; r < sz; ++r) tr[r]= 0;
        for(int l = 0; l < nk-nq;  ++l){
          s[d+outer] = k[l][i][m]*k[l][i][outer];
          tv = v[l][i][outer];
          __syncthreads();
          loc1 = d+rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[loc1+r]*tv;
          }      
        }
        for(int l = 0; l < nq;  ++l){
          s[outer] = q[l][i][m]*q[l][i][outer];
          s[d+outer] = k[nk-nq+l][i][m]*k[nk-nq+l][i][outer];
          tv = v[nk-nq+l][i][outer];
          __syncthreads();
          t = 0;
          loc1 = d+rr*sz;
          loc2 = rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[loc1+r]*tv;
            t += tr[r]*s[loc2+r];
          }      
          o[l][i][outer] += a2*t;
        }
      }
    }

    for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void apply_rpe_and_temp(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rpe_matrix, int bh, int nk, int d, float temperature){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nk; ++l){
      k[l][i][m] /= temperature;
      k[l][i][m] += rpe_matrix[(l+nk)%(2*nk-1)][m];
    }
  }
}

__global__
void calc_norms(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, int bh, int n, int d){
  const int i = threadIdx.x;
  const int l = blockIdx.x;
  float t;
  if(l < n && i < bh){
    t = 0;
    for(int m = 0; m < d; m++){
      t += a[l][i][m]*a[l][i][m];
    }
    norms[l][i] = t;
  }
}

__global__
void find_max(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n){
  const int i = threadIdx.x;
  float t = 0;
  if(i < bh){
    for(int l = 0; l < n; ++l){
      t = max(t,norms[l][i]);
    }
    maxes[i] = t;
  }
}

__global__
void apply_norm(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  float t;
  if(m < d && i < bh){
    t = maxes[i];
    for(int l = 0; l < n; ++l){
      a[l][i][m]*= lim/t;
    }
  }
}

__global__
void apply_dropout(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> drop_noise, float dropout, int bh, int nq, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nq; ++l){
      o[l][i][m] *= (1+dropout*drop_noise[l][i][m]);
    }
  }
}

} // namespace

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor drop_noise,
    torch::Tensor rpe_matrix,
    bool mask,
    float dropout,
    bool normalize,
    float temperature){
    // q: (nq,b*h,d)
    // k: (nk,b*h,d)
    // v: (nk,b*h,d)

  const auto nq = q.size(0);
  const auto nk = k.size(0);
  const auto bh = q.size(1);
  const auto d = q.size(2);

  const int threads = d; // threads = 256
  const int blocks = bh;
  
  auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
  auto o = torch::zeros({nq,bh,d+1},opts);
  auto qnorms = torch::zeros({nq,bh},opts);
  auto knorms = torch::zeros({nk,bh},opts);
  auto qmaxes = torch::zeros({bh},opts);
  auto kmaxes = torch::zeros({bh},opts);

  apply_rpe_and_temp<<<blocks,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rpe_matrix.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nk,d,temperature);

  if(normalize){
    calc_norms<<<nq,bh>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nq,d);
    calc_norms<<<nk,bh>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nk,d);
    find_max<<<1,bh>>>(qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk);
    find_max<<<1,bh>>>(knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq);
    apply_norm<<<blocks,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq,d);
    apply_norm<<<blocks,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk,d);
  }

  if(mask){
    calc_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d);
  }
  else{
    calc_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d);
  }

  cudaDeviceSynchronize();
  apply_dropout<<<blocks,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dropout,bh,nq,d);
  cudaDeviceSynchronize();

  return o;
}