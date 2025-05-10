import torch
import torch.distributed as dist


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()

        # Broadcast parameters from rank 0 to ensure consistency
        for param in self.module.state_dict().values():
            dist.broadcast(param.data, src=0)

        # Register hooks to communicate gradients as they are ready
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param: torch.nn.Parameter):
        def hook(p: torch.Tensor):
            handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)

            def scale_grad(fut):
                p.grad /= self.world_size
                return p.grad

            handle = handle.get_future().then(scale_grad)
            self.handles.append(handle)
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)
