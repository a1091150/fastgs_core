#!/usr/bin/env python3

import os
import sys

import mlx
import mlx.core as mx
import fastgs_core as ext

a = mx.zeros((4, 3), dtype=mx.float32, stream=mx.gpu)

# print("fastgs_core module file:", fastgs_core.__file__)
# print("mlx version:", mx.__version__)
print("mx array type:", type(a))
print("dummy_add:", ext.dummy_add(1, 2))
print("dummy_array_size:", ext.dummy_array_size(a))
