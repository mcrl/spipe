from typing import Optional, Union, Tuple
from collections import deque
from dataclasses import dataclass

import torch

import spiral_helper

global thunder_group
global thunder_cuda_manager


class SpiralBackend:
    def __init__(self, ranks):
        global thunder_group
        thunder_group = spiral_helper.Comm(sorted(ranks))

        global thunder_cuda_manager
        thunder_cuda_manager = SpiralCUDAManager()


def get_thunder_group():
    assert thunder_group is not None, "thunder_group is not initialized"
    return thunder_group


def get_thunder_cuda_manager():
    assert thunder_cuda_manager is not None, "thunder_cuda_manager is not initialized"
    return thunder_cuda_manager


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
        }

        self.__unrecorded_event_hdl_deque = deque()

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
            *self.__stream_dict.get(record_stream_name),
            self.__stream_dict.get(wait_stream_name)[0] if wait_stream_name else None,
            *args,
            **kwargs,
        )
        self.__unrecorded_event_hdl_deque.append(eventhdl)
        return SpiralCUDAEventQuery_t(
            record_stream_name, getattr(eventhdl.event, "spiral_tag")
        )

    def record_event(self, query: SpiralCUDAEventQuery_t) -> int:
        for eventhdl in self.__unrecorded_event_hdl_deque:
            if getattr(eventhdl.event, "spiral_tag") == query.tag:
                eventhdl.record()
                self.__unrecorded_event_hdl_deque.remove(eventhdl)
                return 0
        return -1

    def wait_event(self, query: SpiralCUDAEventQuery_t) -> int:
        for eventhdl in self._get_stream_event_hdl_deque(query.record_stream_name):
            if getattr(eventhdl.event, "spiral_tag") == query.tag:
                eventhdl.wait()
                return 0
        return -1

    def _get_stream_event_hdl_deque(self, stream_name: str):
        assert (
            self.__stream_dict.get(stream_name) is not None
        ), f"Stream {stream_name} is not initialized"
        return self.__stream_dict.get(stream_name)[1]


class SpiralCUDAEventHandle:
    def __init__(
        self,
        record_stream: torch.cuda.Stream,
        record_stream_event_deque: deque,
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
            record_stream_event_deque: the deque to store the event for the record stream
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
        self.record_stream_event_deque = record_stream_event_deque
        self.wait_stream = wait_stream
        self.pre_wait_fn = pre_wait_fn
        self.post_wait_fn = post_wait_fn
        if not hasattr(self.event, "spiral_tag"):
            setattr(self.event, "spiral_tag", tag)

    def record(self):
        """Record the event on the record stream"""
        self.event.record(self.record_stream)
        self.record_stream_event_deque.append(self)

    def wait(self):
        if self.wait_stream is not None:
            assert torch.cuda.current_stream() == self.wait_stream, (
                "The current stream is not predetermined wait stream. "
                "Please make sure to call wait() on the correct stream."
            )

        if self.pre_wait_fn is not None:
            self.pre_wait_fn()

        self.event.wait(stream=torch.cuda.current_stream())

        # flush completed events from the record stream event deque, including the event itself
        # TODO (mcrl): handle case when multiple streams wait on the same event
        while (
            self.record_stream_event_deque
            and self.record_stream_event_deque[0].event.query()
        ):
            completed_event_hdl = self.record_stream_event_deque.popleft()
            if completed_event_hdl == self:
                break

        if self.post_wait_fn is not None:
            self.post_wait_fn()
