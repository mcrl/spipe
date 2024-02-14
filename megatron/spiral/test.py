import nvtx

import torch

from megatron.spiral.debug import spiral_print, spiral_report_memory
from megatron.spiral import get_thunder_cuda_manager


def test_spiral_report_memory():
    """Test spiral_report_memory

    Expected output:
        [Spiral] [0] before | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [2] before | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [3] before | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [1] before | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB

        [Spiral] [0] after cpu alloc | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 2.01 GB
        [Spiral] [1] after cpu alloc | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [2] after cpu alloc | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [3] after cpu alloc | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB

        [Spiral] [0] after gpu transfer | GPU: 0.37 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [1] after gpu transfer | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [2] after gpu transfer | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
        [Spiral] [3] after gpu transfer | GPU: 0.0 GB | MAX_GPU: 0.8 GB | CPU: 1.63 GB
    """

    spiral_report_memory("before", gpu=True, cpu=True)

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        tmp_tensor = torch.zeros(10**8, dtype=torch.float32).cpu()

    torch.distributed.barrier()
    spiral_report_memory("after cpu alloc", gpu=True, cpu=True)

    if torch.distributed.get_rank() == 0:
        tmp_tensor = tmp_tensor.cuda()

    torch.distributed.barrier()
    spiral_report_memory("after gpu transfer", gpu=True, cpu=True)


@nvtx.annotate("test_spiral_cuda_manager")
def test_spiral_cuda_manager():
    TENSOR_SHAPE = torch.Size([10**4, 10**4])
    prefetch_stream = get_thunder_cuda_manager().Stream("prefetch")
    compute_stream = get_thunder_cuda_manager().Stream("compute")

    torch.cuda.nvtx.range_push("setup")
    cpu_tensor = torch.randn(
        TENSOR_SHAPE, dtype=torch.float32, device="cpu", pin_memory=True
    )
    gpu_tensor = torch.zeros(TENSOR_SHAPE, dtype=torch.float32, device="cuda")
    answer = torch.empty(TENSOR_SHAPE, dtype=torch.float32, device="cuda")

    cpu_tensor_clone = torch.clone(cpu_tensor).detach()
    gpu_tensor_verify = cpu_tensor_clone.to(
        "cuda", dtype=torch.float32, non_blocking=False
    )
    verify_answer = torch.empty(TENSOR_SHAPE, dtype=torch.float32, device="cuda")

    multiplier = torch.randn(TENSOR_SHAPE, dtype=torch.float32, device="cuda")

    rand_tensor1 = torch.randn(TENSOR_SHAPE, dtype=torch.float32, device="cuda")
    rand_tensor2 = torch.randn(TENSOR_SHAPE, dtype=torch.float32, device="cuda")
    rand_tensor3 = torch.empty(TENSOR_SHAPE, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # warmups
    torch.cuda.nvtx.range_push("warmup_")
    for _ in range(20):
        torch.add(rand_tensor1, rand_tensor2, out=rand_tensor3)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    with torch.cuda.stream(prefetch_stream):
        prefetch_event_query = get_thunder_cuda_manager().Event(
            "prefetch", "compute", tag="tag:prefetch"
        )
        torch.cuda.nvtx.range_push("copy_")
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        if get_thunder_cuda_manager().record_event(prefetch_event_query) == -1:
            raise RuntimeError("record_event failed")
    with torch.cuda.stream(compute_stream):
        # do some random computation during prefetch
        torch.cuda.nvtx.range_push("random_computation")
        torch.add(rand_tensor1, rand_tensor2, out=rand_tensor3)
        torch.cuda.nvtx.range_pop()

        # wait for prefetch to finish
        if get_thunder_cuda_manager().wait_event(prefetch_event_query) == -1:
            raise RuntimeError("wait_event failed")
        torch.cuda.nvtx.range_pop()

        # do some computation using prefetched data
        torch.cuda.nvtx.range_push("compute_")
        torch.add(gpu_tensor, multiplier, out=answer)
        torch.cuda.current_stream().synchronize()
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("verify")
    torch.add(gpu_tensor_verify, multiplier, out=verify_answer)
    if torch.equal(answer, verify_answer):
        spiral_print("Success")
    else:
        spiral_print("Fail")
    torch.cuda.nvtx.range_pop()
