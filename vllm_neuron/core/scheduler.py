# SPDX-License-Identifier: Apache-2.0
import logging
import os
from collections import deque
from typing import TYPE_CHECKING

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    SchedulerOutput = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronScheduler(Scheduler):
    """Base class inheriting from the V1 scheduler to support continuous
    batching respecting Neuron constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)

        # Requests are temporarily moved to this queue so that the base
        # scheduler does not see them. This lets us ensure that the set of
        # requests scheduled have at least one common warmup shape.
        self.holdback_queue: deque[Request] = deque()

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        """
        Override to prevent assertion crash during speculative decoding.

        Issue: When using EAGLE/speculative decoding near max_model_len,
        the draft model adds extra tokens that can exceed max_model_len,
        causing 'Sampled token IDs exceed the max model length' assertion.

        Fix: Stop generation at (max_model_len - num_spec_tokens) to leave
        room for speculative tokens.
        """
        stopped = False
        spec_len = getattr(self, "num_spec_tokens", 0)
        effective_max_len = max(self.max_model_len - spec_len, 0)

        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            stopped = check_stop(request, effective_max_len)
            if stopped:
                del new_token_ids[num_new:]
                break
        return new_token_ids, stopped


class ContinuousBatchingNeuronScheduler(NeuronScheduler):
    """Support of continuous batching"""

    # inherited from V1 base scheduler but mypy needs to know the type
    running: list[Request]

    def __init__(self, *args, **kwargs) -> None:
        # Initialize NeuronScheduler
        super().__init__(*args, **kwargs)

    def schedule(self) -> "SchedulerOutput":
        """This override adds constraints and then delegates most of the work
        to the base scheduler

        To avoid additional specialization, some requests are held back from the
        base scheduler but are restored after.
        """

        # First purge the full waiting queue into our holdback queue, preserving
        # priority
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # Check if new requests can be scheduled.
        while self.holdback_queue:
            if self.can_schedule(self.holdback_queue[0]):
                # Add request to the waiting queue
                self.waiting.append(self.holdback_queue.popleft())
            else:
                # Otherwise, we simply stop here so that the scheduler
                # can work with the batch we have
                break

        # Schedule Prefill and Decode separately
        if len(self.waiting) > 0:
            # For prefill, hide current decodes from the scheduler
            running_holdback = self.running
            self.running = []
            logger.debug(
                "Scheduling a prefill step of %s requests, holding back %s requests",
                len(self.waiting),
                len(self.holdback_queue),
            )
        else:
            running_holdback = []
            logger.debug("Scheduling a decode step of %s requests", len(self.running))

        # delegate to super of NeuronScheduler: base V1 Scheduler
        outputs = super(NeuronScheduler, self).schedule()

        # restore holdbacks after running the base scheduler
        self.running = self.running + running_holdback
        while self.holdback_queue:
            self.waiting.append(self.holdback_queue.popleft())

        return outputs

    def can_schedule(self, request) -> bool:
        # On Neuron the compiled model processes a fixed batch dimension
        # (max_num_seqs) regardless of how many positions hold real data —
        # unused positions are zero-padded.  Prefilling 1 request at a time
        # wastes (max_num_seqs - 1) positions on padding; batching all
        # waiting requests into a single prefill step is computationally free.
        max_prompt_batch_size = int(
            os.environ.get(
                "NEURON_MAX_PROMPT_BATCH_SIZE",
                str(self.max_num_running_reqs),
            )
        )
        _max_context_len = self.max_model_len

        # running and waiting queues are both empty -> start new batch
        start_new_batch = len(self.running) + len(self.waiting) == 0
        # check that there is space in the current decode batch
        cond1 = len(self.running) + len(self.waiting) < self.max_num_running_reqs
        # check that there is space in the prefill batch
        cond2 = len(self.waiting) < max_prompt_batch_size
        # TODO: add context length checks
        return start_new_batch or (cond1 and cond2)
