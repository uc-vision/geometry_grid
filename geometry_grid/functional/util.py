from contextlib import contextmanager
import torch

@contextmanager
def restore_grad(*tensors):
  try:
      grads = [tensor.grad for tensor in tensors]
      yield
  finally:
      for tensor, grad in zip(tensors, grads):
          tensor.grad = grad


@contextmanager
def clear_grad(*tensors):
  try:
      yield
  finally:
      for tensor in tensors:
          tensor.grad = None

