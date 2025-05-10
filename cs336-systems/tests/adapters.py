#!/usr/bin/env python3
from __future__ import annotations

from typing import Type
import torch
from torch.autograd import Function
from ddp_overlap import DDP, BucketedDDP, ShardedOptimizer

def get_rmsnorm_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm.
    The expectation is that this class will implement RMSNorm
    using standard PyTorch operations.

    Returns:
        A class object (not an instance of the class)
    """
    class RMSNormForwardPyTorch(Function):
        @staticmethod
        def forward(ctx, x, weight, eps=1e-5):
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
            return (x / rms) * weight

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            eps = ctx.eps
            H = x.shape[-1]

            rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
            normed_x = x / rms

            grad_weight = torch.sum(grad_output * normed_x, dim=tuple(range(x.ndim - 1)))
            dx_norm = grad_output * weight

            mean_dx_norm_x = (dx_norm * x).mean(dim=-1, keepdim=True)
            grad_x = (dx_norm - normed_x * mean_dx_norm_x / (rms ** 2)) / rms

            return grad_x, grad_weight, None

    return RMSNormForwardPyTorch


def get_rmsnorm_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_rmsnorm_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    import triton
    import triton.language as tl

    @triton.jit
    def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, eps, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + offsets, mask=mask, other=1.0)
        rms = tl.sqrt(tl.sum(x * x, axis=0) / n_elements + eps)
        y = (x / rms) * w
        tl.store(y_ptr + offsets, y, mask=mask)

    class RMSNormForwardTriton(Function):
        @staticmethod
        def forward(ctx, x, weight, eps=1e-5):
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            x = x.contiguous()
            weight = weight.contiguous()
            y = torch.empty_like(x)

            B = x.numel() // x.shape[-1]
            H = x.shape[-1]
            grid = lambda meta: (B,)
            rmsnorm_fwd_kernel[grid](
                x_ptr=x,
                w_ptr=weight,
                y_ptr=y,
                eps=eps,
                n_elements=H,
                BLOCK_SIZE=H,
            )
            return y

        @staticmethod
        def backward(ctx, grad_output):
            raise NotImplementedError("Triton backward not implemented")

    return RMSNormForwardTriton


def rmsnorm_backward_g(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    grad_g = torch.sum(grad_output * (x / rms), dim=tuple(range(x.ndim - 1)))
    return grad_g


def rmsnorm_backward_x_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    H = x.shape[-1]
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    normed_x = x / rms

    dx_norm = grad_output * g
    mean_dx_norm_x = (dx_norm * x).mean(dim=-1, keepdim=True)
    grad_x = (dx_norm - normed_x * mean_dx_norm_x / (rms ** 2)) / rms
    return grad_x


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    return DDP(module)


def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    return BucketedDDP(module, bucket_size_mb=bucket_size_mb)


def ddp_bucketed_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    pass  # Optional, depending on implementation


def get_sharded_optimizer(
    params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
