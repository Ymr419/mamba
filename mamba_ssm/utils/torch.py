import torch
from functools import partial
from typing import Callable

try:
    import torch_npu  # noqa: F401  # Ascend NPU backend; provides torch.npu / torch.npu.amp
    _npu_available = True
except ImportError:
    _npu_available = False

# torch_npu exposes multi_processor_count as None (E13); the real core-count
# attributes on Ascend devices are cube_core_num / vector_core_num.
amp_device_type = "npu" if _npu_available else "cuda"

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = amp_device_type
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    if _npu_available:
        from torch.npu.amp import custom_fwd, custom_bwd
    else:
        from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)
