#include "sampling.h"
#include "bvh.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_point_sampling", &furthest_point_sampling);
  m.def("furthest_point_sampling_with_mask", &furthest_point_sampling_with_mask);
  m.def("ray_tracing", &bvh_ray_tracing, "BVH Ray Tracing forward (CUDA)",
          py::arg("triangles"), py::arg("points"), py::arg("directions"),
          py::arg("queue_size") = 128,
          py::arg("sort_points_by_morton") = true);
}
