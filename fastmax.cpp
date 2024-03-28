#include <iostream>
#include <vector>
#include <math.h>
#include <torch/extension.h>
#include <iostream>
using namespace std;

// Forward Pass
vector<torch::Tensor> forwardpass(torch::Tensor qq, torch::Tensor kk, torch::Tensor vv, int mask = 1, int p = 2){
    // q and k should be noralized in the Python wrapper, as well as applying the denum_term
    auto x = qq.sizes();
    int b = x[0];
    int h = x[1];
    int nq = x[2];
    int nk = kk.sizes()[2];
    int d = x[3];

    auto q = qq.packed_accessor32<float,4>();
    auto k = kk.packed_accessor32<float,4>();
    auto v = vv.packed_accessor32<float,4>();
    // auto options =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
    // vector<vector<vector<vector<float>>>> o;
    // vector<vector<vector<float>>> denum;
    // vector<vector<vector<vector<float>>>> zero (b,vector<vector<vector<float>>>(h, vector<vector<float>>(nq, vector<float>(d, 0.0))));
    // o = zero;
    // vector<vector<vector<float>>> zerod (b,vector<vector<float>>(h, vector<float>(nq,  0.0)));
    // denum = zerod;
    vector<float> o (b*h*nq*d, 0.0);
    vector<float> denum (b*h*nq, 0.0);

    float denum_term = 1.0;
    float denum_term2 = 0.5;
    float cons;
    vector<float> lin (d, 0.0);
    vector<vector<float>> quad (d, vector<float>(d, 0.0));
    int index_d, index_o;

    if(mask == 0){
        //calc denum terms
        for(int i = 0; i < b; i++){
            for(int j = 0; j < h; j++){
                cons = 0;
                vector<float> lin (d, 0.0);
                vector<vector<float>> quad (d, vector<float>(d, 0.0));
                for(int l = 0; l < nk; l++){
                    cons += 1;
                    for(int m = 0; m < d; m++){
                        lin[m] += denum_term*k[i][j][l][m];
                        if(p == 2) {
                            for (int r = 0; r < d; r++) {
                                quad[m][r] += denum_term2*k[i][j][l][r]*k[i][j][l][m];
                            }
                        }
                    }
                }
                for(int l = 0; l < nq; l++){
                    index_d = i*h*nq + j*nq + l;
                    denum[index_d] = cons;
                    for(int m = 0; m < d; m++){
                        denum[index_d] += q[i][j][l][m]*lin[m];
                        if(p == 2) {
                            for (int r = 0; r < d; r++) {
                                denum[index_d] += q[i][j][l][m]*q[i][j][l][r]*quad[m][r];
                            }
                        }
                    }
                }
            }
        }

        //calc num term
        for(int outer = 0; outer < d; outer++){
            for(int i = 0; i < b; i++){
                for(int j = 0; j < h; j++){
                    cons = 0;
                    vector<float> lin (d, 0.0);
                    vector<vector<float>> quad (d, vector<float>(d, 0.0));
                    for(int l = 0; l < nk; l++){
                        cons += v[i][j][l][outer];
                        for(int m = 0; m < d; m++){
                            lin[m] += denum_term*k[i][j][l][m]*v[i][j][l][outer];
                            if(p == 2) {
                                for (int r = 0; r < d; r++) {
                                    quad[m][r] += denum_term2*k[i][j][l][r]*k[i][j][l][m]*v[i][j][l][outer];
                                }
                            }
                        }
                    }
                    for(int l = 0; l < nq; l++){
                        index_d = i*h*nq + j*nq + l;
                        index_o = index_d*d + outer;
                        o[index_o] = cons;
                        for(int m = 0; m < d; m++){
                            o[index_o] += q[i][j][l][m]*lin[m];
                            if(p == 2) {
                                for (int r = 0; r < d; r++) {
                                    o[index_o] += q[i][j][l][m]*q[i][j][l][r]*quad[m][r];
                                }
                            }
                        }
                        o[index_o] /= denum[index_d]; //element-wise div
                    }
                }
            }
        }
    }
    else{
        int n = nq;
        //calc denum terms
        for(int i = 0; i < b; i++){
            for(int j = 0; j < h; j++){
                float cons = 0;
                vector<float> lin (d, 0.0);
                vector<vector<float>> quad (d, vector<float>(d, 0.0));
                for(int l = 0; l < n; l++){                    
                    index_d = i*h*nq + j*nq + l;
                    cons += 1;
                    for(int m = 0; m < d; m++){
                        lin[m] += denum_term*k[i][j][l][m];
                        if(p == 2){
                            for(int r = 0; r < d; r++){
                                quad[m][r] += denum_term2*k[i][j][l][r]*k[i][j][l][m];
                            }
                        }
                    }
                    denum[index_d] = cons;
                    for(int m = 0; m < d; m++){
                        denum[index_d] += q[i][j][l][m]*lin[m];
                        if(p == 2) {
                            for (int r = 0; r < d; r++) {
                                denum[index_d] += q[i][j][l][m]*q[i][j][l][r]*quad[m][r];
                            }
                        }
                    }
                }
            }
        }

        //calc num terms
        for(int outer = 0; outer < d; outer++){
            for(int i = 0; i < b; i++){
                for(int j = 0; j < h; j++){
                    float cons = 0;
                    vector<float> lin (d, 0.0);
                    vector<vector<float>> quad (d, vector<float>(d, 0.0));
                    for(int l = 0; l < n; l++){                        
                        index_d = i*h*nq + j*nq + l;
                        index_o = index_d*d + outer;
                        cons += v[i][j][l][outer];
                        for(int m = 0; m < d; m++){
                            lin[m] += denum_term*k[i][j][l][m]*v[i][j][l][outer];
                            if(p == 2){
                                for(int r = 0; r < d; r++){
                                    quad[m][r] += denum_term2*k[i][j][l][r]*k[i][j][l][m]*v[i][j][l][outer];
                                }
                            }
                        }
                        o[index_o] = cons;
                        for(int m = 0; m < d; m++){
                            o[index_o] += q[i][j][l][m]*lin[m];
                            if(p == 2) {
                                for (int r = 0; r < d; r++){
                                    o[index_o] += q[i][j][l][m]*q[i][j][l][r]*quad[m][r];
                                }
                            }
                        }
                        o[index_o] /= denum[index_d]; //element-wise div
                    }
                }
            }
        }
    }
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor oo = torch::from_blob(o.data(), {b,h,nq,d}, opts).clone();
    torch::Tensor ddenum = torch::from_blob(denum.data(), {b,h,nq}, opts).clone();
    return {oo,ddenum};
}

//#######################################################################################################################
// Backward Pass
torch::Tensor backwardpass(torch::Tensor qq, torch::Tensor kk, torch::Tensor vv, torch::Tensor oo, torch::Tensor denumm, torch::Tensor grad_outputt, int mask = 1, int p = 2){
    // q and k should be noralized in the Python wrapper, as well as applying the denum_term
    auto x = qq.sizes();
    int b = x[0];
    int h = x[1];
    int nq = x[2];
    int nk = kk.sizes()[2];
    int d = x[3];

    auto q = qq.packed_accessor32<float,4>();
    auto k = kk.packed_accessor32<float,4>();
    auto v = vv.packed_accessor32<float,4>();
    auto o = oo.packed_accessor32<float,4>();
    auto denum = denumm.packed_accessor32<float,3>();
    auto grad_output = grad_outputt.packed_accessor32<float,4>();

    // vector<vector<vector<vector<vector<float>>>>> grad; //grad[0] = grad_q, grad[1] = grad_k, grad[2] = grad_v
    // vector<vector<vector<vector<float>>>> zeroq (b,vector<vector<vector<float>>>(h, vector<vector<float>>(nq, vector<float>(d, 0.0))));
    // grad[0] = zeroq;
    // vector<vector<vector<vector<float>>>> zerok (b,vector<vector<vector<float>>>(h, vector<vector<float>>(nk, vector<float>(d, 0.0))));
    // grad[1] = zerok;
    // grad[2] = zerok;

    vector<float> grad (b*h*nq*d + b*h*nk*d + b*h*nk*d, 0.0);
    int index;


    vector<vector<float>> gradq_coeffs1v (d, vector<float>(d, 0));
    vector<vector<float>> gradq_coeffs1o (d, vector<float>(d, 0));
    vector<vector<float>> gradk_coeffs1v (d, vector<float>(d, 0));
    vector<vector<float>> gradk_coeffs1o (d, vector<float>(d, 0));
    vector<vector<float>> gradv_coeffs2 (d, vector<float>(d, 0));
    vector<vector<float>> k2 (d, vector<float>(d, 0));

    if(mask == 0) {
        // grad terms for q
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < h; j++) {
                for (int outer = 0; outer < d; outer++) {
                    vector<float> gradq_coeffs0v(d, 0);
                    vector<float> gradq_coeffs0o(d, 0);
                    if (p == 2) {
                        vector<vector<float>> gradq_coeffs1v(d, vector<float>(d, 0));
                        vector<vector<float>> gradq_coeffs1o(d, vector<float>(d, 0));
                    }
                    for (int l = 0; l < nk; l++) {
                        for (int m = 0; m < d; m++) {
                            gradq_coeffs0v[m] += v[i][j][l][outer] * k[i][j][l][m];
                            gradq_coeffs0o[m] += k[i][j][l][m];
                            if (p == 2) {
                                for (int mm = 0; mm < d; mm++) {
                                    gradq_coeffs1v[m][mm] += v[i][j][l][outer] * k[i][j][l][m] * k[i][j][l][mm];
                                    gradq_coeffs1o[m][mm] += k[i][j][l][m] * k[i][j][l][mm];
                                }
                            }
                        }
                    }
                    for (int l = 0; l < nq; l++) {
                        for (int m = 0; m < d; m++) {
                            index = i*h*nq*d + j*nq*d + l*d + m;
                            grad[index] += (gradq_coeffs0v[m] - gradq_coeffs0o[m] * o[i][j][l][outer]) *
                                                   grad_output[i][j][l][outer] / denum[i][j][l];
                            if (p == 2) {
                                for (int mm = 0; mm < d; mm++) {
                                    grad[index] +=
                                            (gradq_coeffs1v[m][mm] - gradq_coeffs1o[m][mm] * o[i][j][l][outer]) *
                                            q[i][j][l][mm] * grad_output[i][j][l][outer] / denum[i][j][l];
                                }
                            }
                        }
                    }
                }
            }
        }

        // grad terms for k
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < h; j++) {
                for (int outer = 0; outer < d; outer++) {
                    vector<float> gradk_coeffs0v(d, 0);
                    vector<float> gradk_coeffs0o(d, 0);
                    if (p == 2) {
                        vector<vector<float>> gradk_coeffs1v(d, vector<float>(d, 0));
                        vector<vector<float>> gradk_coeffs1o(d, vector<float>(d, 0));
                    }
                    for (int l = 0; l < nq; l++) {
                        for (int m = 0; m < d; m++) {
                            gradk_coeffs0v[m] += q[i][j][l][m] * grad_output[i][j][l][outer] / denum[i][j][l];
                            gradk_coeffs0o[m] +=
                                    o[i][j][l][outer] * q[i][j][l][m] * grad_output[i][j][l][outer] / denum[i][j][l];
                            if (p == 2) {
                                for (int mm = 0; mm < d; mm++) {
                                    gradk_coeffs1v[m][mm] +=
                                            q[i][j][l][mm] * q[i][j][l][m] * grad_output[i][j][l][outer] /
                                            denum[i][j][l];;
                                    gradk_coeffs1o[m][mm] += o[i][j][l][outer] * q[i][j][l][mm] * q[i][j][l][m] *
                                                             grad_output[i][j][l][outer] / denum[i][j][l];
                                }
                            }
                        }
                    }
                    for (int l = 0; l < nk; l++) {
                        for (int m = 0; m < d; m++) {                            
                            index = b*h*nq*d + i*h*nk*d + j*nk*d + l*d + m;
                            grad[index] += gradk_coeffs0v[m] * v[i][j][l][outer] - gradk_coeffs0o[m];
                            if (p == 2) {
                                for (int mm = 0; mm < d; mm++) {
                                    grad[index] +=
                                            (gradk_coeffs1v[m][mm] * v[i][j][l][outer] - gradq_coeffs1o[m][mm]) *
                                            k[i][j][l][mm];
                                }
                            }
                        }
                    }
                }
            }
        }

        // grad terms for v
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < h; j++) {
                for (int outer = 0; outer < d; outer++) {
                    float gradv_coeffs0 = 0;
                    vector<float> gradv_coeffs1(d, 0);
                    if (p == 2) vector<vector<float>> gradv_coeffs2(d, vector<float>(d, 0));
                    for (int l = 0; l < nq; l++) {
                        gradv_coeffs0 += grad_output[i][j][l][outer] / denum[i][j][l];
                        for (int m = 0; m < d; m++) {
                            gradv_coeffs1[m] += q[i][j][l][m] * grad_output[i][j][l][outer] / denum[i][j][l];
                            for (int mm = 0; mm < d; mm++) {
                                gradv_coeffs2[m][mm] +=
                                        0.5 * q[i][j][l][m] * q[i][j][l][mm] * grad_output[i][j][l][outer] /
                                        denum[i][j][l];
                            }
                        }
                    }
                    for (int l = 0; l < nk; l++) {
                        for (int m = 0; m < d; m++) {
                            index = b*h*(nq+nk)*d + i*h*nk*d + j*nk*d + l*d + m;
                            grad[index] = gradv_coeffs0;
                        }
                        for (int m = 0; m < d; m++) {
                            index = b*h*(nq+nk)*d + i*h*nk*d + j*nk*d + l*d + m;
                            grad[index] += gradv_coeffs1[m] * k[i][j][l][m];
                            if (p == 2) {
                                for (int mm = 0; mm < d; mm++) {
                                    grad[index] += gradv_coeffs2[m][mm] * k[i][j][l][m] * k[i][j][l][mm];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    else{
        // grad terms for q
        for(int i = 0; i < b; i++){
            for(int j = 0; j < h; j++){
                for(int outer = 0; outer < d; outer++){
                    vector<float> gradq_coeffs0v (d, 0);
                    vector<float> gradq_coeffs0o (d, 0);
                    if(p == 2){
                        vector<vector<float>> gradq_coeffs1v (d, vector<float>(d, 0));
                        vector<vector<float>> gradq_coeffs1o (d, vector<float>(d, 0));
                    }
                    for(int l = 0; l < nk; l++) {
                        for (int m = 0; m < d; m++) {
                            gradq_coeffs0v[m] += v[i][j][l][outer] * k[i][j][l][m];
                            gradq_coeffs0o[m] += k[i][j][l][m];
                            if (p == 2) {
                                for (int mm = 0; mm < d; mm++) {
                                    gradq_coeffs1v[m][mm] += v[i][j][l][outer] * k[i][j][l][m] * k[i][j][l][mm];
                                    gradq_coeffs1o[m][mm] += k[i][j][l][m] * k[i][j][l][mm];
                                }
                            }
                        }

                        for (int m = 0; m < d; m++) {
                            index = i*h*nq*d + j*nq*d + l*d + m;
                            grad[index] += (gradq_coeffs0v[m] - gradq_coeffs0o[m] * o[i][j][l][outer]) *
                                                   grad_output[i][j][l][outer] / denum[i][j][l];
                            if (p == 2) {
                                for (int mm = 0; mm < d; mm++) {
                                    grad[index] +=
                                            (gradq_coeffs1v[m][mm] - gradq_coeffs1o[m][mm] * o[i][j][l][outer]) *
                                            q[i][j][l][mm] * grad_output[i][j][l][outer] / denum[i][j][l];
                                }
                            }
                        }
                    }
                }
            }
        }

        // grad terms for k
        for(int i = 0; i < b; i++){
            for(int j = 0; j < h; j++){
                for(int outer = 0; outer < d; outer++){
                    vector<float> gradk_coeffs0v (d, 0);
                    vector<float> gradk_coeffs0o (d, 0);
                    if(p == 2){
                        vector<vector<float>> gradk_coeffs1v (d, vector<float>(d, 0));
                        vector<vector<float>> gradk_coeffs1o (d, vector<float>(d, 0));
                    }
                    for(int l = nq-1; l >= 0; l--){
                        for(int m = 0; m < d; m++){
                            gradk_coeffs0v[m] += q[i][j][l][m]*grad_output[i][j][l][outer]/denum[i][j][l];
                            gradk_coeffs0o[m] += o[i][j][l][outer]*q[i][j][l][m]*grad_output[i][j][l][outer]/denum[i][j][l];
                            if(p == 2){
                                for(int mm = 0; mm < d; mm++){
                                    gradk_coeffs1v[m][mm] += q[i][j][l][mm]*q[i][j][l][m]*grad_output[i][j][l][outer]/denum[i][j][l];;
                                    gradk_coeffs1o[m][mm] += o[i][j][l][outer]*q[i][j][l][mm]*q[i][j][l][m]*grad_output[i][j][l][outer]/denum[i][j][l];
                                }
                            }
                        }

                        for(int m = 0; m < d; m++){
                            index = b*h*nq*d + i*h*nk*d + j*nk*d + l*d + m;
                            grad[index] += gradk_coeffs0v[m]*v[i][j][l][outer] - gradk_coeffs0o[m];
                            if(p == 2){
                                for(int mm = 0; mm < d; mm++){
                                    grad[index] += (gradk_coeffs1v[m][mm]*v[i][j][l][outer] - gradq_coeffs1o[m][mm])*k[i][j][l][mm];
                                }
                            }
                        }
                    }
                }
            }
        }

        // grad terms for v
        for(int i = 0; i < b; i++){
            for(int j = 0; j < h; j++) {
                for (int outer = 0; outer < d; outer++) {
                    float gradv_coeffs0 = 0;
                    vector<float> gradv_coeffs1 (d, 0);
                    if(p == 2) vector<vector<float>> gradv_coeffs2 (d, vector<float>(d, 0));
                    for(int l = nq - 1; l >= 0; l--){
                        gradv_coeffs0 += grad_output[i][j][l][outer]/denum[i][j][l];
                        for(int m = 0; m < d; m++){
                            gradv_coeffs1[m] += q[i][j][l][m]*grad_output[i][j][l][outer]/denum[i][j][l];
                            for(int mm = 0; mm < d; mm++){
                                gradv_coeffs2[m][mm] += 0.5*q[i][j][l][m]*q[i][j][l][mm]*grad_output[i][j][l][outer]/denum[i][j][l];
                            }
                        }

                        for(int m = 0; m < d; m++) {
                            index = b*h*(nq+nk)*d + i*h*nk*d + j*nk*d + l*d + m;
                            grad[index] = gradv_coeffs0;
                        }
                        for(int m = 0; m < d; m++){
                            index = b*h*(nq+nk)*d + i*h*nk*d + j*nk*d + l*d + m;
                            grad[index] += gradv_coeffs1[m]*k[i][j][l][m];
                            if(p == 2){
                                for(int mm = 0; mm < d; mm++){
                                    grad[index] += gradv_coeffs2[m][mm]*k[i][j][l][m]*k[i][j][l][mm];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor ggrad = torch::from_blob(grad.data(), {3,b,h,nq,d}, opts).clone();
    return ggrad;
}



PYBIND11_MODULE(fastmax_cpu, m) {
  m.def("forwardpass", &forwardpass, "forwardpass");
  m.def("backwardpass", &backwardpass, "backwardpass");
}