import os

import torch
from torch import nn
import spipe_helper
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
        thunder_group = spipe_helper.Comm(sorted(ranks))


    def test_borrow_tensor(self):
        fillvalue = self.rank
        thunder_group.SetSpiralCPUAllocator()
        t = torch.full((2, 2), fillvalue, dtype=torch.float, requires_grad=False, device='cpu')
        print(f"[{self.rank}] {t.data_ptr()} {t.data}")

        sys.stdout.flush()
        self.comm.Barrier()

        if self.rank == 0:
            f = int(input("Type src rank: "))
            a = int(input("Type src addr: "))

            thunder_group.BorrowTensor(t, a, f)

            print(f"[{self.rank}] borrowed tensor from {f} {t.data_ptr()} {t.data}")

        self.comm.Barrier()


    def test_borrow_module(self):
        thunder_group.SetSpiralCPUAllocator()

        m = nn.Linear(2, 2, bias=False, device='cpu' if self.rank != 0 else 'meta')
        print(f"[{self.rank}] {m.weight} ({m.weight.data_ptr()})")
        sys.stdout.flush()
        self.comm.Barrier()

        if self.rank == 0:
            f = int(input("Type src rank: "))
            a = int(input("Type src addr: "))

            m.to_empty(device='cpu')
            print(f"[{self.rank}] {m.weight} ({m.weight.data_ptr()})")

            thunder_group.BorrowTensor(m.weight, a, f)

            print(f"[{self.rank}] {m.weight} ({m.weight.data_ptr()})")

        self.comm.Barrier()


    def test_offload(self):
        spipe_helper.LazyConfigure(True)
        thunder_group.SetSpiralCPUAllocator()
        tensor_ = torch.empty(10, 10, dtype=torch.float, requires_grad=False, device='cpu')
        print(tensor_)

        thunder_group.UnsetSpiralCPUAllocator()
        tensor_ = torch.empty(10, 10, dtype=torch.float, requires_grad=False, device='cpu')
        print(tensor_)


if __name__ == '__main__':
    sprl = SpiralBackend()

    sprl.test_offload()
    # sprl.test_borrow_tensor()
    # sprl.test_borrow_module()