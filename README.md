# SPipe

Hybrid GPU and CPU Pipeline for Training LLMs under Memory Pressure

## Introduction

SPipe is a LLM training framework that enables efficient utilization of multiple GPUs and CPUs for training under limited compute and memory resources. SPipe presents two key techniques for pipeline parallelism (PP) to efficiently leverage CPU offloading: (1) Decoupled Pass Assignment and (2) Asynchronous CPU Optimizer. 

(1) stores model parameters in CPU memory rather than GPU memory, and assigns forward and backward passes to two separate GPUs that access the shared CPU-resident parameters. This design effectively reduces inter-GPU pipeline bubbles that arise in conventional settings where a single GPU performs both passes. (2) performs the GPU backward passes and CPU optimizer steps for the already-completed parameters in parallel. This parallelism alleviates pipeline bubbles that occur when there is no overlap between GPU and CPU processing. Additionally, SPipe provides a set of supporting techniques: Fine-grained Stage Partitioning to eliminate inter-GPU pipeline bubbles, Asynchronous Checkpoint Communication to optimize GPU communication, and Bypassing and Rollback mechanisms to ensure the efficiency and correctness of the asynchronous optimizer.

## Repository Organization

```
spipe/
├── csrc/                   # SPipe C++ package
│  ├── spipe_helper/        #    - SHMEM+RDMA backend
│  ├── spipe_cpu_adam/      #    - CPU optimizer
│  ├── common/              #    - Utility functions
│  ├── external/            #    - External libraries
│
├── megatron/               # SPipe Python package
│  ├── spipe/               #    - Pipeline schedule
│
├── external/               # External libraries
│
└── examples/               # SPipe usage examples
```

## Build

TBD

## Examples

We provide working examples for running SPipe in the [`examples/`](/examples) directory.

## Artifact Evaluation

See [Artifact Evaluation](./AE.md) for the details to reproduce the results in the [paper]().

## References
If you find SPipe relevant to your research, please consider citing:
```
@inproceedings{spipe-pact25,
    title = {{SPipe}: Hybrid {GPU} and {CPU} Pipeline for Training {LLMs} under Memory Pressure},
    author = {Ryu, Junyeol and Jeong, Yujin and Park, Daeyoung and Kim, Jinpyo and Kim, Heehoon and Lee, Jaejin},
    booktitle = {Proceedings of the 34th ACM/IEEE/IFIP International Conference on Parallel Architectures and Compilation Techniques},
    year = {2025},
}
```

## Contact

Junyeol Ryu (jyeol.ryu@gmail.com)