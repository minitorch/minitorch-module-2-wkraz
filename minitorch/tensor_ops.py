from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(fn: Callable[[float], float]) -> Any:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # loop over all indices of out tensor
        for out_index in range(len(out)):
            # convert the flat tensor to a multidimensional index w/ our function we defined earlier
            out_multidim_index = [0] * len(out_shape)
            to_index(out_index, out_shape, out_multidim_index)
            
            # map the multidimensional index to the input tensor
            in_multidim_index = [0] * len(in_shape)
            broadcast_index(out_multidim_index, out_shape, in_shape, in_multidim_index)
            
            # convert the multidimensional index to a flat tensor
            in_flat_index = index_to_position(in_multidim_index, in_strides)
            
            out[out_index] = fn(in_storage[in_flat_index])
            
        

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        for out_index in range(len(out)):
            # convert flat index to multidimensional index
            out_multidim_index = [0] * len(out_shape)
            to_index(out_index, out_shape, out_multidim_index)
            
            # map the multidim index to tensor a
            a_multidim_index = [0] * len(a_shape)
            broadcast_index(out_multidim_index, out_shape, a_shape, a_multidim_index)
            
            # map it to b now
            b_multidim_index = [0] * len(b_shape)
            broadcast_index(out_multidim_index, out_shape, b_shape, b_multidim_index)
            
            # convert the multidim indices to flat indices
            a_flat_index = index_to_position(a_multidim_index, a_strides)
            b_flat_index = index_to_position(b_multidim_index, b_strides)
            
            # apply fn to values a and b in storage (with flat indices since that's how they're stored)
            out[out_index] = fn(a_storage[a_flat_index], b_storage[b_flat_index])

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.zeros(len(out_shape)).astype(int)
        for i in range(len(out)):
            # convert flat index to multidim index
            to_index(i, out_shape, out_index)
            
            # iterate over all elements in reduced dimension
            for j in range(a_shape[reduce_dim]):
                out_index_temp = out_index[reduce_dim]      # temporary
                out_index[reduce_dim] = j
                # convert multidim index to flat
                pos = index_to_position(out_index, a_strides)
                # apply reduction function to value in a_storage and accumulate the result in out[i]
                out[i] = fn(a_storage[pos], out[i])
                out_index[reduce_dim] = out_index_temp      # restore original value
    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
