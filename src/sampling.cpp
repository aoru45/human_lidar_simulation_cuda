#include "sampling.h"
#include "utils.h"

void furthest_point_sampling_kernel_wrapper(int b, int n, int c, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

void furthest_point_sampling_xue_kernel_wrapper(int b, int n, int c, int m,
                                            const float *dataset, const bool* mask, float *temp,
                                            int *idxs);



at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), nsamples, points.data_ptr<float>(),
        tmp.data_ptr<float>(), output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

at::Tensor furthest_point_sampling_with_mask(at::Tensor points, at::Tensor mask, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(mask);
  CHECK_IS_FLOAT(points);
  CHECK_IS_BOOL(mask);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    furthest_point_sampling_xue_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), nsamples, points.data_ptr<float>(),
        mask.data_ptr<bool>(),tmp.data_ptr<float>(), output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
