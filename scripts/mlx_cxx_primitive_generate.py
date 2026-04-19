#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADER_TEMPLATE = """#pragma once

#include <tuple>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace fastgs_core {{
namespace mx = mlx::core;

struct {class_name}Params {{
  // TODO: fill params
}};

struct {class_name}Input {{
  mx::array input;
  mx::StreamOrDevice s;
  {class_name}Params params;
}};

enum {class_name}OutputIndex {{
  kOutput = 0,
}};

std::vector<mx::array> {func_name}(const {class_name}Input& input);

class {class_name} : public mx::Primitive {{
 public:
  {class_name}(mx::Stream stream, {class_name}Params params)
      : mx::Primitive(stream), params_(params) {{}}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& tangents,
                             const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& cotangents,
                             const std::vector<int>& argnums,
                             const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {{ return "{class_name}"; }}

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  {class_name}Params params_;
}};

}}  // namespace fastgs_core
"""

CPP_TEMPLATE = """#include "include/{file_stem}.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "helper.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace fastgs_core {{

std::vector<mx::array> {func_name}(const {class_name}Input& input) {{
  auto prim = std::make_shared<{class_name}>(to_stream(input.s), input.params);

  const int n = input.input.shape(0);

  std::vector<mx::Shape> output_shapes = {{
      {{n}},
  }};
  std::vector<mx::Dtype> output_types = {{
      input.input.dtype(),
  }};

  std::vector<mx::array> inputs = {{
      mx::contiguous(input.input),
  }};

  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}}

#ifdef _METAL_
void {class_name}::eval_gpu(const std::vector<mx::array>& inputs,
                            std::vector<mx::array>& outputs) {{
  for (auto& out : outputs) {{
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }}

  auto& input = inputs[0];
  auto& output = outputs[0];

  const int n = input.shape(0);

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("fastgs_core", current_binary_dir());
  auto kernel = d.get_kernel("{func_name}_forward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_bytes(n, 0);
  compute_encoder.set_input_array(input, 1);
  compute_encoder.set_output_array(output, 2);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(n, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}}
#else
void {class_name}::eval_gpu(const std::vector<mx::array>&,
                            std::vector<mx::array>&) {{
  throw std::runtime_error("{class_name} has no GPU implementation.");
}}
#endif

void {class_name}::eval_cpu(const std::vector<mx::array>&,
                            std::vector<mx::array>& outputs) {{
  for (auto& out : outputs) {{
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }}
}}

std::vector<mx::array> {class_name}::jvp(const std::vector<mx::array>&,
                                         const std::vector<mx::array>&,
                                         const std::vector<int>&) {{
  throw std::runtime_error("{class_name} jvp is not implemented.");
}}

std::vector<mx::array> {class_name}::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {{
  throw std::runtime_error("{class_name} vjp is not implemented.");
}}

std::pair<std::vector<mx::array>, std::vector<int>> {class_name}::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {{
  throw std::runtime_error("{class_name} vmap is not implemented.");
}}

bool {class_name}::is_equivalent(const mx::Primitive& other) const {{
  return name() == other.name();
}}

}}  // namespace fastgs_core
"""

METAL_TEMPLATE = """#include <metal_stdlib>
using namespace metal;

kernel void {func_name}_forward_kernel(
    constant int& n [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {{
  if (tid >= static_cast<uint>(n)) {{
    return;
  }}

  output[tid] = input[tid];
}}
"""


def to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def validate_class_name(name: str) -> None:
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name):
        raise ValueError(
            "Class name must be PascalCase, e.g. Foo or FastGSPreprocess"
        )


def write_file(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        print(f"[skip] {path} already exists")
        return

    path.write_text(content, encoding="utf-8")
    print(f"[ok]   wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a primitive header/cpp/metal skeleton."
    )
    parser.add_argument("class_name", help="PascalCase class name, e.g. Foo")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()

    class_name = args.class_name.strip()
    validate_class_name(class_name)

    file_stem = to_snake(class_name)
    func_name = file_stem

    repo_root = Path(__file__).resolve().parent.parent

    header_path = repo_root / "fastgs_core" / "include" / f"{file_stem}.h"
    cpp_path = repo_root / "fastgs_core" / f"{file_stem}.cpp"
    metal_path = repo_root / "fastgs_core" / "metal" / f"{file_stem}.metal"

    replacements = {
        "class_name": class_name,
        "file_stem": file_stem,
        "func_name": func_name,
    }

    write_file(header_path, HEADER_TEMPLATE.format(**replacements), args.force)
    write_file(cpp_path, CPP_TEMPLATE.format(**replacements), args.force)
    write_file(metal_path, METAL_TEMPLATE.format(**replacements), args.force)

    print()
    print("Generated names:")
    print(f"  class_name : {class_name}")
    print(f"  file_stem  : {file_stem}")
    print(f"  func_name  : {func_name}")


if __name__ == "__main__":
    main()
