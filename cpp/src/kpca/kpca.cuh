/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <cuml/matrix/kernelparams.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/transpose.h>
#include <cuml/common/device_buffer.hpp>
#include <cuml/decomposition/params.hpp>
#include <matrix/kernelfactory.cuh>
#include <matrix/kernelmatrices.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <stats/cov.cuh>
#include <tsvd/tsvd.cuh>
#include <common/device_utils.cuh>
#include <rmm/device_vector.hpp>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/host/allocator.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace ML {

using namespace MLCommon;

/**
 * @brief perform fit operation for the pca. Generates eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[out] mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @param[out] noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void kpcaFit(const raft::handle_t &handle, math_t *input, math_t *components,
             math_t *explained_var, math_t *explained_var_ratio,
             math_t *singular_vals, math_t *mu, math_t *noise_vars,
             const paramsPCA &prms, cudaStream_t stream) {
  auto cublas_handle = handle.get_cublas_handle();
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");

  int n_components = prms.n_components;
  if (n_components > prms.n_rows) n_components = prms.n_rows;

  raft::print_device_vector("input matrix (as vector): ", input,
                            prms.n_rows * prms.n_cols, std::cout);

  //   Matrix::KernelParams kparam{Matrix::RBF, 0, 1, 0};
  Matrix::KernelParams kparam{Matrix::LINEAR, 0, 0, 0};
  Matrix::GramMatrixBase<math_t> *kernel =
    Matrix::KernelFactory<math_t>::create(kparam, cublas_handle);
  
  math_t *kernel_mat;
  raft::allocate(kernel_mat, prms.n_rows * prms.n_rows);
  kernel->evaluate(input, prms.n_rows, prms.n_cols, input, prms.n_rows,
         kernel_mat, false, stream, prms.n_rows, prms.n_rows, prms.n_rows);
  
  raft::print_device_vector("kernel matrix (as vector): ", kernel_mat,
                            prms.n_rows * prms.n_rows, std::cout);

  //  center the kernel matrix 
  //  K' = (I - 1n) K (I - 1n)
  math_t * i_mat;
  raft::allocate(i_mat, prms.n_rows * prms.n_rows);

  math_t * diag;
  raft::allocate(diag, prms.n_rows);
  auto thrust_policy = rmm::exec_policy(stream);
  thrust::fill(thrust_policy, diag, diag + prms.n_rows, 1.0f);
  math_t inv_n_rows = 1.0/prms.n_rows;

  math_t * centering_mat;
  raft::allocate(centering_mat, prms.n_rows * prms.n_rows);

  raft::matrix::initializeDiagonalMatrix(diag, i_mat, prms.n_rows, prms.n_rows, stream);

  raft::print_device_vector("identity matrix: ", i_mat,
                            prms.n_rows * prms.n_rows, std::cout);

  raft::linalg::subtractScalar(centering_mat, i_mat, inv_n_rows, prms.n_rows * prms.n_rows, stream);

  raft::print_device_vector("centering_mat: ", centering_mat,
                            prms.n_rows * prms.n_rows, std::cout);


  math_t * temp_mat;
  raft::allocate(temp_mat, prms.n_rows * prms.n_rows);
  math_t alpha = 1.0f;
  math_t beta = 0.0f;
  
  raft::linalg::gemm(handle, centering_mat, prms.n_rows, prms.n_rows
              , kernel_mat, temp_mat, prms.n_rows, prms.n_rows
              , CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);


  raft::print_device_vector("temp_mat: ", temp_mat,
                            prms.n_rows * prms.n_rows, std::cout);

  math_t * centered_kernel;
  raft::allocate(centered_kernel, prms.n_rows * prms.n_rows);

  raft::linalg::gemm(handle, temp_mat, prms.n_rows, prms.n_rows
              , centering_mat, centered_kernel, prms.n_rows, prms.n_rows
              , CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);
  
  raft::print_device_vector("centered_kernel: ", centered_kernel,
                            prms.n_rows * prms.n_rows, std::cout);

  math_t tol = 1.e-7;
  math_t sweeps = 15;
  //  eigendecomposition
  raft::linalg::eigJacobi(handle, centered_kernel, prms.n_rows,
                     prms.n_rows, components, explained_var,
                     stream, tol, sweeps);

  raft::print_device_vector("components: ", components,
                            prms.n_rows * n_components, std::cout);
  raft::print_device_vector("explained_var (eigenvalues): ", explained_var,
                            n_components, std::cout);
  
  
  //  scale eigenvectors
  //  sort eigenvectors and eigenvalues
  //  truncate zero eigenvectors/eigenvalues
  //  

  //  scale alphas
  std::cout << "END KPCA FIT\n";
}

/**
 * @brief perform fit and transform operations for the pca. Generates transformed data, eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @param[out] trans_input: the transformed data. Size n_rows * n_components.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[out] mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @param[out] noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void kpcaFitTransform(const raft::handle_t &handle, math_t *input,
                      math_t *trans_input, math_t *components,
                      math_t *explained_var, math_t *explained_var_ratio,
                      math_t *singular_vals, math_t *mu, math_t *noise_vars,
                      const paramsPCA &prms, cudaStream_t stream) {
  kpcaFit(handle, input, components, explained_var, explained_var_ratio,
          singular_vals, mu, noise_vars, prms, stream);
  kpcaTransform(handle, input, components, trans_input, singular_vals, mu, prms,
                stream);
  signFlip(trans_input, prms.n_rows, prms.n_components, components, prms.n_cols,
           handle.get_device_allocator(), stream);
}

/**
 * @brief performs transform operation for the pca. Transforms the data to eigenspace.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: the data is transformed. Size n_rows x n_components.
 * @param[in] components: principal components of the input data. Size n_cols * n_components.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] singular_vals: singular values of the data. Size n_components * 1.
 * @param[in] mu: mean value of the input data
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void kpcaTransform(const raft::handle_t &handle, math_t *input,
                   math_t *components, math_t *trans_input,
                   math_t *singular_vals, math_t *mu, const paramsPCA &prms,
                   cudaStream_t stream) {
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 0,
         "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");

  std::cout << "IN KPCA TRANSFORM \n";
  raft::stats::meanCenter(input, input, mu, prms.n_cols, prms.n_rows, false,
                          true, stream);

  tsvdTransform(handle, input, components, trans_input, prms, stream);
  raft::stats::meanAdd(input, input, mu, prms.n_cols, prms.n_rows, false, true,
                       stream);
  std::cout << "KPCA TRANSFORM DONE \n";
}

};  // end namespace ML
