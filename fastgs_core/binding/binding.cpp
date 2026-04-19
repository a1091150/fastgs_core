#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "../include/dummy.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_fastgs_core, m) {
  m.def("dummy_add", &fastgs_core::dummy_add, "a"_a, "b"_a);
}
