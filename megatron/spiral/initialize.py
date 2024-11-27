from typing import Optional, Union, Tuple
from collections import deque
from dataclasses import dataclass
import atexit
import os
import psutil
import subprocess

import torch

import spiral_helper

from megatron import get_args
from megatron.spiral.debug import spiral_print

SPIRAL_BACKEND = None


class SpiralBackend:
    def __init__(
        self,
        ranks,
        device,
        init_shmem,
        shared_memory_name,
        shared_memory_buffer_size,
        shared_memory_header_size,
        alignment,
    ):
        self._set_numa() # NOTE: Currently does nothing but added to later pass numa ID directly to Comm
        self._set_cpu_affinity()
        self.thunder_group = spiral_helper.Comm(
            sorted(ranks),
            device,
            init_shmem,
            shared_memory_name,
            shared_memory_buffer_size,
            shared_memory_header_size,
            alignment,
            psutil.Process().cpu_num(),
        )
        self.thunder_cuda_manager = SpiralCUDAManager()
        global SPIRAL_BACKEND
        SPIRAL_BACKEND = self

        # NOTE (SpiralPipe) Below enforces invocation of destructor for all objects in SpiralBackend when the program exits, either normally or abnormally. This is especially critical for spiral_helper.Comm, which allocates a hugh shared memory.
        atexit.register(self.__del__)

    def _set_numa(self):
        cpu_numa_bindings = {}
        lscpu_output = subprocess.check_output(["lscpu", "--parse=CPU,NODE"]).decode()
        for line in lscpu_output.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            cpu, node = map(int, line.split(","))
            cpu_numa_bindings[cpu] = node
        self.numa = {cpu_numa_bindings[cpu] for cpu in psutil.Process().cpu_affinity()}
        spiral_print(f"SpiralBackend numa nodes: {self.numa}")

    def _set_cpu_affinity(self):
        # NOTE: StageOptimizer requires separting cores for main and optimizer threads
        affinity_cores = psutil.Process().cpu_affinity()
        if get_args().spiral_stage_optimizer:
            curr_cpu_num = psutil.Process().cpu_num()
            hyper_cpu_num = curr_cpu_num + 32 if curr_cpu_num < 32 else curr_cpu_num - 32 # TODO: this is hardcoded for 64-core machine
            os.sched_setaffinity(0, {curr_cpu_num, hyper_cpu_num})
            self.cpu_affinity = [curr_cpu_num, hyper_cpu_num]
            self.available_cpus = list(filter(lambda x: x != curr_cpu_num and x != hyper_cpu_num, affinity_cores))
        else:
            self.cpu_affinity = affinity_cores
            self.available_cpus = affinity_cores
        spiral_print(f"SpiralBackend cpu affinity: {self.cpu_affinity}")

    def __del__(self):
        global SPIRAL_BACKEND
        if SPIRAL_BACKEND is not None:
            SPIRAL_BACKEND = None


def get_thunder_group():
    global SPIRAL_BACKEND
    assert SPIRAL_BACKEND is not None, "SpiralBackend is not initialized"
    return SPIRAL_BACKEND.thunder_group


def get_thunder_cuda_manager():
    global SPIRAL_BACKEND
    assert SPIRAL_BACKEND is not None, "SpiralBackend is not initialized"
    return SPIRAL_BACKEND.thunder_cuda_manager


def get_available_cpus():
    global SPIRAL_BACKEND
    assert SPIRAL_BACKEND is not None, "SpiralBackend is not initialized"
    return SPIRAL_BACKEND.available_cpus


@dataclass(frozen=True)
class SpiralCUDAEventQuery_t:
    record_stream_name: str
    tag: Union[str, int]


class SpiralCUDAManager:
    def __init__(self):
        assert (
            torch.cuda.current_device() >= 0
        ), "SpiralCUDAManager should be initialized after torch.cuda.set_device() is called"

        self.__stream_dict: dict[
            str, Tuple[torch.cuda.Stream, deque[SpiralCUDAEventHandle]]
        ] = {
            "compute": (
                torch.cuda.Stream(torch.cuda.current_device(), priority=0),
                deque(),
            ),
            "prefetch": (
                torch.cuda.Stream(torch.cuda.current_device(), priority=0),
                deque(),
            ),
            "offload": (
                torch.cuda.Stream(torch.cuda.current_device(), priority=0),
                deque(),
            ),
            "free": (
                torch.cuda.Stream(torch.cuda.current_device(), priority=0),
                deque(),
            ),
        }

        self.__unrecorded_event_hdl_deque = deque()
        self.__completed_event_hdl_deque = (
            deque()
        )  # sorted new ~ old for efficient query

    def Stream(self, stream_name: str):
        assert (
            self.__stream_dict.get(stream_name) is not None
        ), f"Stream {stream_name} is not initialized"
        return self.__stream_dict.get(stream_name)[0]

    def Event(
        self, record_stream_name: str, wait_stream_name: Optional[str], *args, **kwargs
    ) -> SpiralCUDAEventQuery_t:
        assert (
            self.__stream_dict.get(record_stream_name) is not None
        ), f"Stream {record_stream_name} is not initialized"
        if wait_stream_name is not None:
            assert (
                self.__stream_dict.get(wait_stream_name) is not None
            ), f"Stream {wait_stream_name} is not initialized"
        eventhdl = SpiralCUDAEventHandle(
            self.__stream_dict.get(record_stream_name)[0],
            self.__stream_dict.get(wait_stream_name)[0] if wait_stream_name else None,
            *args,
            **kwargs,
        )
        self.__unrecorded_event_hdl_deque.append(eventhdl)
        return SpiralCUDAEventQuery_t(
            record_stream_name, getattr(eventhdl.event, "spiral_tag")
        )

    def record_event(self, query: SpiralCUDAEventQuery_t) -> int:
        """Record event and return cuda_event identifier (pylong)"""
        _target_stream, _target_event_hdl_deque = self._get_stream_event_hdl_deque(
            query.record_stream_name
        )
        for eventhdl in self.__unrecorded_event_hdl_deque:
            if getattr(eventhdl.event, "spiral_tag") == query.tag:
                assert eventhdl.record_stream == _target_stream
                eventhdl.record()
                self.__unrecorded_event_hdl_deque.remove(eventhdl)
                _target_event_hdl_deque.append(eventhdl)
                return eventhdl.event.cuda_event
        return -1

    def wait_event(self, query: SpiralCUDAEventQuery_t, sync=False) -> int:
        _target_event_hdl_deque = self._get_stream_event_hdl_deque(
            query.record_stream_name
        )[1]
        for eventhdl in _target_event_hdl_deque:
            if getattr(eventhdl.event, "spiral_tag") == query.tag:
                if sync:
                    eventhdl.synchronize()
                else:
                    eventhdl.wait()
                _target_event_hdl_deque.remove(eventhdl)
                self.__completed_event_hdl_deque.appendleft(eventhdl)
                return 0
        # Query event from completed list if not found
        for eventhdl in self.__completed_event_hdl_deque:
            if getattr(eventhdl.event, "spiral_tag") == query.tag:
                if sync:
                    eventhdl.synchronize()
                else:
                    eventhdl.wait()
                return 0
        return -1

    def get_event(self, query: SpiralCUDAEventQuery_t) -> Optional[torch.cuda.Event]:
        _target_event_hdl_deque = self._get_stream_event_hdl_deque(
            query.record_stream_name
        )[1]
        for eventhdl in _target_event_hdl_deque:
            if getattr(eventhdl.event, "spiral_tag") == query.tag:
                return eventhdl.event
        # Query event from completed list if not found
        for eventhdl in self.__completed_event_hdl_deque:
            if getattr(eventhdl.event, "spiral_tag") == query.tag:
                return eventhdl.event
        return None

    def _get_stream_event_hdl_deque(self, stream_name: str):
        assert (
            self.__stream_dict.get(stream_name) is not None
        ), f"Stream {stream_name} is not initialized"
        return self.__stream_dict.get(stream_name)

    def __repr__(self) -> str:
        _str = ""
        for stream_name, (stream, event_hdl_deque) in self.__stream_dict.items():
            _str += f"{stream_name}({len(event_hdl_deque)}) "
        return _str

    def __del__(self):
        for stream_name, (stream, event_hdl_deque) in self.__stream_dict.items():
            if len(event_hdl_deque) > 0:
                print(
                    f"WARNING: {len(event_hdl_deque)} unwaited events on {stream_name} stream"
                )
        if len(self.__unrecorded_event_hdl_deque) > 0:
            print(
                f"WARNING: {len(self.__unrecorded_event_hdl_deque)} unrecorded events"
            )


class SpiralCUDAEventHandle:
    def __init__(
        self,
        record_stream: torch.cuda.Stream,
        wait_stream: Optional[torch.cuda.Stream],
        pre_wait_fn: Optional[callable] = None,
        post_wait_fn: Optional[callable] = None,
        tag: Optional[Union[str, int]] = None,
        *args,
        **kwargs,
    ):
        """Wrapper class for torch.cuda.Event with additional functionalities

        An event can be recorded on a single predetermined stream, but can be waited on multiple streams.
        Arguments:
            record_stream: the stream on which the event will be recorded
            wait_stream: the stream on which the event will be waited. If None, any stream can wait on the event

        Keyword Arguments:
            pre_wait_fn: the function to be called before waiting on the event
            post_wait_fn: the function to be called after waiting on the event
            tag: the tag to be attached to the event
        """
        self.event = torch.cuda.Event(
            *args, **kwargs
        )  # NOTE: `blocking` argument of torch.cuda.Event ctor is whether to
        # wait by polling-based or event-driven, so we do not care about it here.
        # https://github.com/pytorch/pytorch/issues/82061
        self.record_stream = record_stream
        self.wait_stream = wait_stream
        self.pre_wait_fn = pre_wait_fn
        self.post_wait_fn = post_wait_fn
        if not hasattr(self.event, "spiral_tag"):
            setattr(self.event, "spiral_tag", tag)

    def record(self):
        """Record the event on the record stream"""
        self.event.record(self.record_stream)

    def wait(self):
        if self.wait_stream is not None:
            assert torch.cuda.current_stream() == self.wait_stream, (
                "The current stream is not predetermined wait stream. "
                "Please make sure to call wait() on the correct stream."
            )

        if self.pre_wait_fn is not None:
            self.pre_wait_fn()

        self.event.wait(stream=torch.cuda.current_stream())

        if self.post_wait_fn is not None:
            self.post_wait_fn()

    def synchronize(self):
        assert (
            self.wait_stream is None
        ), "EventSynchronize would block cpu processing, so wait_stream should be None"

        if self.pre_wait_fn is not None:
            self.pre_wait_fn()

        self.event.synchronize()

        if self.post_wait_fn is not None:
            self.post_wait_fn()