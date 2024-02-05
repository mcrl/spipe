import torch

from megatron.spiral.debug import spiral_print, spiral_report_memory


def test_spiral_report_memory():
    """ Test spiral_report_memory 
    
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

    spiral_report_memory('before', gpu=True, cpu=True)

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        tmp_tensor = torch.zeros(10**8, dtype=torch.float32).cpu()
    
    torch.distributed.barrier()
    spiral_report_memory('after cpu alloc', gpu=True, cpu=True)

    if torch.distributed.get_rank() == 0:
        tmp_tensor = tmp_tensor.cuda()
    
    torch.distributed.barrier()
    spiral_report_memory('after gpu transfer', gpu=True, cpu=True)