// main.cpp
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include <raft/handle.hpp>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/allocator.hpp>
#include <raft/linalg/transpose.h>

#include <cuml/cluster/kmeans.hpp>

#include <cuml/decomposition/params.hpp>
#include <cuml/decomposition/kpca.hpp>
#include <cuml/matrix/kernelparams.h>

#include "./csv.cpp"
#include "./timelogger.cpp"
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
  }
#endif  // CUDA_RT_CALL

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg,
             const T default_val) {
  T argval = default_val;
  char **itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

int main(int argc, char *argv[]) {
  printf("1AFTER UPDATE \n");
  int devId = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  // int cudaStatus = cudaSetDevice(devId);
   {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaSetDevice(devId);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not select CUDA device with the id: " << devId
                << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
    cudaStatus = cudaFree(0);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not initialize CUDA on device: " << devId
                << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
  }
  raft::handle_t handle;
  std::shared_ptr<raft::mr::device::allocator> allocator(
    new raft::mr::device::default_allocator());

  handle.set_device_allocator(allocator);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  handle.set_stream(stream);
  //  load data file
  std::string dataset = "iris";
  std::string filename = dataset + ".csv";
  std::cout << "Here\n" << " /home/tomas/618/kernelpca/data/" << filename << std::endl;
  csvInfo csv = read_csv("/home/tomas/618/kernelpca/data/" + filename);
  std::cout << "matrix[0][0] " << csv.matrix[0] << ", size " << csv.matrix.size() << std::endl;
  std::cout << "rows " << csv.rows << " cols " << csv.cols << std::endl;

  // //  number of principal components to find
  int n_components = 4;
  TimeLogger *tl = new TimeLogger(csv.rows, csv.cols, n_components, "/home/tomas/618/kernelpca/logs/CUMLSKPCA_" + filename);
  ML::paramsKPCA prms;
  float* data = nullptr;
  float* data_tranposed = nullptr;
  float *trans_data = nullptr;
  float *alphas = nullptr;
  float *lambdas = nullptr;
  int len = csv.rows * csv.cols;
  CUDA_RT_CALL(cudaMalloc(&data,
                            len * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(&data_tranposed,
                            len * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(&trans_data,
                            len * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(&alphas,
                            csv.rows * csv.rows * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(&lambdas,
                            csv.rows * sizeof(float)));
  CUDA_RT_CALL(cudaMemcpyAsync(data, csv.matrix.data(),
                                 (size_t) len * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
  float *alphas_h;
  float *lambdas_h;
  float *trans_data_h;
  alphas_h = (float*) malloc(csv.rows * n_components * sizeof(float));
  lambdas_h = (float*) malloc(n_components * sizeof(float));
  trans_data_h = (float*) malloc(n_components * csv.rows * sizeof(float));
  raft::linalg::transpose(handle, data, data_tranposed, csv.cols, csv.rows, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  // raft::update_device(data, csv.matrix.data(), len, stream);
  raft::print_device_vector("data: ", data,
                              csv.cols * 10, std::cout);
  raft::print_device_vector("data_tranposed: ", data_tranposed,
                              csv.cols * 10, std::cout);
  prms.n_cols = csv.cols;
  prms.n_rows = csv.rows;
  prms.n_components = n_components;
  prms.algorithm = ML::solver::COV_EIG_JACOBI;
  std::vector<std::pair<std::string, MLCommon::Matrix::KernelParams>> kernels;
  kernels.push_back(std::make_pair("RBF",MLCommon::Matrix::KernelParams{MLCommon::Matrix::RBF, 0, (float)1.0/(float)csv.rows, 1.0}));
  kernels.push_back(std::make_pair("LINEAR",MLCommon::Matrix::KernelParams{MLCommon::Matrix::LINEAR, 0, 0.0, 0.0}));
  kernels.push_back(std::make_pair("POLYNOMIAL",MLCommon::Matrix::KernelParams{MLCommon::Matrix::POLYNOMIAL, 3, (float)1.0/(float)csv.rows, 1}));
  TimeLogger::timeLog *total_time = tl->start("Total Time");
  for(int i = 0; i < kernels.size(); i++) {
    std::pair<std::string, MLCommon::Matrix::KernelParams> kpair = kernels[i];
    std::string kernel = kpair.first;
    prms.kernel = kpair.second;
    std::cout << "prms.n_cols " << prms.n_cols << std::endl;
    std::cout << "degree " << prms.kernel.degree << std::endl;
    std::cout << "gamma " << prms.kernel.gamma << std::endl;
    std::cout << "coef0 " << prms.kernel.coef0 << std::endl;
    std::cout << "kernel " << prms.kernel.kernel << std::endl;
    // kpcaFit(handle, data_tranposed, alphas, lambdas, prms);
    // std::cout << "kpcaFit Done" << std::endl;
    // kpcaTransform(handle, data_tranposed, alphas, lambdas, trans_data, prms);
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    TimeLogger::timeLog *curr_kernel = tl->start(kernel);
    kpcaFitTransform(handle, data_tranposed, trans_data, alphas, lambdas, prms);
    tl->stop(curr_kernel);
    printf("CUML KPCA Kernel %s on file %s TOTAL Time measured: %f ms.\n", kernel.c_str(), filename.c_str(), curr_kernel->time_ms);
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    // raft::print_device_vector("alphas: ", alphas,
    //                             csv.rows * n_components, std::cout);
    raft::print_device_vector("lambdas: ", lambdas,
                                n_components, std::cout);
    // raft::print_device_vector("trans_data: ", trans_data,
    //                             n_components * csv.rows, std::cout);

    CUDA_RT_CALL(cudaMemcpyAsync(alphas_h, alphas,
                                  (size_t) csv.rows * n_components * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaMemcpyAsync(lambdas_h, lambdas,
                                  n_components * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaMemcpyAsync(trans_data_h, trans_data,
                                  n_components * sizeof(float) * csv.rows,
                                  cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    std::string outFilePath = "/home/tomas/618/kernelpca/output/";
    write_matrix_csv(outFilePath + "CUMLSKPCA_" + dataset + "_" + kernel + "_lambdas.csv", lambdas_h, n_components, 1);
    write_matrix_csv(outFilePath + "CUMLSKPCA_" + dataset + "_" + kernel + "_alphas.csv", alphas_h, n_components, csv.rows);
    write_matrix_csv(outFilePath + "CUMLSKPCA_" + dataset + "_" + kernel + "_trans_data.csv", trans_data_h, n_components, csv.rows);
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
  }
  tl->stop(total_time);
  printf("CUML KPCA on file %s TOTAL Time measured: %f ms.\n", filename.c_str(), total_time->time_ms);
  return 0;
}
