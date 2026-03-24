#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
      auto current_batch_size = std::min((size_t)batch, m - i);

      auto X_batch = X + i * n;
      auto y_batch = y + i;

      std::vector<float> logits(current_batch_size * k);
      std::vector<float> prob(current_batch_size * k);
      std::vector<float> gradient(n * k);

      for (size_t row = 0; row < current_batch_size; row++) {
        for (size_t col = 0; col < k; col++) {
          auto dot_result = 0.0f;
          for (size_t i = 0; i < n; i++) {
            dot_result += X_batch[row * n + i] * theta[i * k + col];
          }
          logits[row * k + col] = dot_result;
        }
      }

      for (size_t row = 0; row < current_batch_size; row++) {
        auto sum = 0.0f;

        for (size_t col = 0; col < k; col++) {
          sum += exp(logits[row * k + col]);
        }

        for (size_t col = 0; col < k; col++) {
          prob[row * k + col] = exp(logits[row * k + col]) / sum;
        }
      }

      for (size_t row = 0; row < current_batch_size; row++) {
        prob[row * k + y_batch[row]] -= 1.0f;
      }

      for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < k; col++) {
          auto res = 0.0f;
          for (size_t i = 0; i < current_batch_size; i++) {
            res += X_batch[i * n + row] * prob[i * k + col];
          }
          gradient[row * k + col] = res / current_batch_size;
        }
      }

      for (size_t i = 0; i < n * k; i++){
        theta[i] -= lr * gradient[i];
      }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
