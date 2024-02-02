import os

import torch
import spiral_helper
from mpi4py import MPI
import sys

global thunder_group

class SpiralBackend:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()

        ranks = [i for i in range(self.world_size)]
        global thunder_group
        thunder_group = spiral_helper.Comm(sorted(ranks))


    def test_borrow_tensor(self):
        fillvalue = self.rank
        thunder_group.SetSpiralCPUAllocator()
        t = torch.full((2, 2), fillvalue, dtype=torch.float, requires_grad=False, device='cpu')
        print(f"[{self.rank}] {t.data_ptr()}")

        sys.stdout.flush()
        self.comm.Barrier()

        if self.rank == 0:
            f = int(input("Type src rank: "))
            a = int(input("Type src addr: "))

            thunder_group.BorrowTensor(t, a, f)

            print(t)

        self.comm.Barrier()
        

    def test_offload(self):
        spiral_helper.lazy_configure(True)
        thunder_group.SetSpiralCPUAllocator()    
        tensor_ = torch.empty(10, 10, dtype=torch.float, requires_grad=False, device='cpu')
        print(tensor_)

        thunder_group.UnsetSpiralCPUAllocator()
        tensor_ = torch.empty(10, 10, dtype=torch.float, requires_grad=False, device='cpu')
        print(tensor_)


if __name__ == '__main__':
    sprl = SpiralBackend()

    # sprl.test_offload()
    sprl.test_borrow_tensor()