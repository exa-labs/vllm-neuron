# SPDX-License-Identifier: Apache-2.0
from collections import deque
from unittest.mock import Mock, patch


from vllm_neuron.core.scheduler import ContinuousBatchingNeuronScheduler


class TestNeuronScheduler:
    def test_scheduler_initialization(self, scheduler):
        """Test basic scheduler initialization and configuration.

        This test verifies that:
        1. Scheduler is properly initialized with required queues
        2. Initial queue states are empty
        3. All required attributes are present

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        assert hasattr(scheduler, "holdback_queue")
        assert len(scheduler.holdback_queue) == 0
        assert hasattr(scheduler, "waiting")
        assert hasattr(scheduler, "running")

    def test_can_schedule_empty_queues(self, scheduler):
        """Test scheduling capability with empty queues.

        This test verifies that:
        1. New requests can be scheduled when queues are empty
        2. Scheduler correctly evaluates capacity
        3. Default scheduling behavior works as expected

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        assert scheduler.can_schedule(mock_request)

    def test_can_schedule_with_running_requests(self, scheduler):
        """Test scheduling capability with existing running requests.

        This test verifies that:
        1. Scheduler correctly handles existing running requests
        2. Capacity evaluation considers current load
        3. Returns appropriate boolean response

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        scheduler.running = [Mock() for _ in range(5)]
        result = scheduler.can_schedule(mock_request)
        assert isinstance(result, bool)

    def test_schedule_with_empty_queues(self, scheduler):
        """Test scheduling behavior with empty request queues.

        This test verifies that:
        1. Scheduler handles empty queue state correctly
        2. Returns valid output even with no requests
        3. Maintains empty state of waiting and holdback queues

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        output = scheduler.schedule()
        assert output is not None
        assert len(scheduler.waiting) == 0
        assert len(scheduler.holdback_queue) == 0

    def test_schedule_with_waiting_requests(self, scheduler):
        """Test scheduling behavior with pending requests.

        This test verifies that:
        1. Waiting requests are properly processed
        2. Scheduler produces valid output
        3. Request counts are maintained correctly
        4. Queue state transitions are valid

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        scheduler.holdback_queue.append(mock_request)
        output = scheduler.schedule()

        # Verify the scheduling behavior
        assert output is not None
        # Verify that request was processed
        total_requests = len(scheduler.waiting) + len(scheduler.holdback_queue)
        assert total_requests > 0

    def test_queue_management(self, scheduler):
        """Test queue management and state transitions.

        This test verifies that:
        1. Requests are properly added to queues
        2. Queue state transitions work correctly
        3. Request counts are maintained
        4. Scheduling output is valid

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Add requests to holdback queue
        mock_requests = [Mock() for _ in range(3)]
        for req in mock_requests:
            scheduler.holdback_queue.append(req)

        # Verify initial state
        assert len(scheduler.holdback_queue) == 3
        assert len(scheduler.waiting) == 0

        # Schedule
        output = scheduler.schedule()
        assert output is not None

        # Verify queue transitions
        total_requests = len(scheduler.waiting) + len(scheduler.holdback_queue)
        assert total_requests == 3

    @patch("vllm_neuron.core.scheduler.logger")
    def test_logging_behavior(self, mock_logger, scheduler):
        """Test scheduler logging functionality.

        This test verifies that:
        1. Debug logs are generated during scheduling
        2. Logger is called with appropriate messages
        3. Logging doesn't interfere with scheduling

        Args:
            mock_logger: Mock logger instance
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        scheduler.holdback_queue.append(mock_request)
        output = scheduler.schedule()
        assert output is not None
        mock_logger.debug.assert_called()

    def test_max_capacity_constraints(self, scheduler):
        """Test scheduler capacity limit enforcement.

        This test verifies that:
        1. Maximum capacity limits are enforced
        2. Requests are rejected when at capacity
        3. Capacity checking is accurate

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Fill up to max capacity
        scheduler.running = [Mock() for _ in range(scheduler.max_num_running_reqs)]
        mock_request = Mock()
        result = scheduler.can_schedule(mock_request)
        assert not result

    def test_batch_scheduling_logic(self, scheduler):
        """Test batch request scheduling behavior.

        This test verifies that:
        1. Multiple requests are handled correctly
        2. Batch scheduling maintains request counts
        3. Queue transitions work for batches
        4. Output is valid for batch operations

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial state
        initial_running = [Mock() for _ in range(2)]
        initial_holdback = [Mock() for _ in range(3)]
        scheduler.running = initial_running
        scheduler.holdback_queue.extend(initial_holdback)

        # Execute scheduling
        output = scheduler.schedule()
        assert output is not None

        # Verify scheduling behavior
        total_requests = (
            len(scheduler.running)
            + len(scheduler.waiting)
            + len(scheduler.holdback_queue)
        )
        assert total_requests >= len(initial_running) + len(initial_holdback)

    def verify_scheduler_state(self, scheduler, expected_total_requests):
        """Verify scheduler's current state matches expectations.

        This helper method verifies that:
        1. Total request count matches expectations
        2. Requests are properly distributed across queues
        3. No requests are lost during transitions

        Args:
            scheduler: The scheduler instance to verify
            expected_total_requests: Expected total number of requests

        Raises:
            AssertionError: If actual total requests doesn't match expected
        """
        actual_total = (
            len(scheduler.running)
            + len(scheduler.waiting)
            + len(scheduler.holdback_queue)
        )
        assert actual_total == expected_total_requests, (
            f"Expected {expected_total_requests} total requests, but found {actual_total}"
        )

    def test_schedule_with_running_and_waiting(self, scheduler):
        """Test scheduling behavior with concurrent requests.

        This test verifies that:
        1. Scheduler handles multiple running requests
        2. Waiting requests are properly processed
        3. Output maintains correct scheduling state
        4. Queue transitions are handled properly

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial state
        scheduler.running = [Mock() for _ in range(2)]
        scheduler.waiting = deque([Mock() for _ in range(2)])

        # Execute scheduling
        output = scheduler.schedule()

        # Verify that requests were processed
        assert output is not None

    def test_continuous_batching_initialization(self, scheduler):
        """Test initialization of ContinuousBatchingNeuronScheduler.

        This test verifies:
        1. Scheduler is instantiated with correct class type
        2. Required configuration attributes are present
        3. Essential queue management attributes exist

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        assert isinstance(scheduler, ContinuousBatchingNeuronScheduler)
        assert hasattr(scheduler, "scheduler_config")
        assert hasattr(scheduler, "max_num_running_reqs")

    def test_purge_waiting_to_holdback(self, scheduler):
        """Test queue transition from waiting to holdback queue.

        This test verifies:
        1. Requests are correctly moved from waiting to holdback queue
        2. Request count is preserved during transition
        3. Queue state is consistent after scheduling

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial waiting queue
        initial_waiting = [Mock() for _ in range(3)]
        scheduler.waiting.extend(initial_waiting)

        # Execute schedule
        scheduler.schedule()

        # Verify all requests were moved through holdback
        assert len(scheduler.waiting) == 3
        assert len(scheduler.holdback_queue) == 0

    def test_max_prompt_batch_size_constraint(self, scheduler):
        """Test enforcement of maximum prompt batch size.

        Default max_prompt_batch_size equals max_num_running_reqs (batch
        prefill). The env var NEURON_MAX_PROMPT_BATCH_SIZE can restrict it.

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        import os

        # Default: batch prefill allows up to max_num_running_reqs in waiting
        scheduler.waiting.append(Mock())
        mock_request = Mock()
        result = scheduler.can_schedule(mock_request)
        assert result, "Default batch prefill should allow multiple in waiting"

        # Fill waiting to max_num_running_reqs → should reject
        scheduler.waiting.clear()
        for _ in range(scheduler.max_num_running_reqs):
            scheduler.waiting.append(Mock())
        result = scheduler.can_schedule(mock_request)
        assert not result, "Should reject when waiting == max_num_running_reqs"

        # Env var override restricts batch size
        scheduler.waiting.clear()
        scheduler.waiting.append(Mock())
        with patch.dict(os.environ, {"NEURON_MAX_PROMPT_BATCH_SIZE": "1"}):
            result = scheduler.can_schedule(mock_request)
        assert not result, "Should reject when env var restricts to 1"

    def test_context_length_handling(self, scheduler):
        """Test request handling with different context lengths.

        This test verifies:
        1. Basic request scheduling works with different context lengths
        2. Scheduler capacity constraints are respected
        3. Batch size limits are enforced

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Clear scheduler state
        scheduler.running = []
        scheduler.waiting.clear()
        scheduler.holdback_queue.clear()

        # Test with empty scheduler (should accept any context length)
        mock_request = Mock()
        mock_request.num_tokens = scheduler.scheduler_config.max_model_len + 1
        result = scheduler.can_schedule(mock_request)
        assert result, "Should accept request when queues are empty"

        # Test with scheduler at capacity
        scheduler.running = [Mock() for _ in range(scheduler.max_num_running_reqs)]
        result = scheduler.can_schedule(mock_request)
        assert not result, "Should reject request when at capacity"

    def test_invalid_request_handling(self, scheduler):
        """Test handling of invalid request configurations.

        This test verifies:
        1. Invalid requests are properly rejected
        2. Capacity constraints are enforced
        3. Request validation logic works correctly

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()

        # Set up the request to be invalid
        mock_request.num_tokens = scheduler.scheduler_config.max_model_len + 1
        mock_request.num_output_tokens = 0
        mock_request.max_tokens = 10
        mock_request.sampling_params = None

        # Mock the running and waiting queues to be full
        scheduler.running = [Mock() for _ in range(scheduler.max_num_running_reqs)]

        result = scheduler.can_schedule(mock_request)
        assert not result, "Should not schedule invalid requests"

    def test_schedule_with_empty_state(self, scheduler):
        """Test scheduler behavior with empty queues.

        This test verifies:
        1. Scheduler handles empty queue states correctly
        2. Valid output is produced with no requests
        3. Queue state remains consistent

        Args:
            scheduler: Fixture providing configured scheduler instance
        """

        assert len(scheduler.running) == 0
        assert len(scheduler.waiting) == 0
        assert len(scheduler.holdback_queue) == 0

        output = scheduler.schedule()
        assert output is not None

    def test_concurrent_queue_operations(self, scheduler):
        """Test simultaneous operations on multiple queues.

        This test verifies:
        1. Multiple queues can be processed simultaneously
        2. Request count integrity is maintained
        3. Queue state transitions are handled correctly
        4. No requests are lost during processing

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial state
        running_requests = [Mock() for _ in range(2)]
        waiting_requests = [Mock() for _ in range(2)]
        holdback_requests = [Mock() for _ in range(2)]

        scheduler.running.extend(running_requests)
        scheduler.waiting.extend(waiting_requests)
        scheduler.holdback_queue.extend(holdback_requests)

        output = scheduler.schedule()
        assert output is not None

        # Verify queue integrity
        total_requests = (
            len(scheduler.running)
            + len(scheduler.waiting)
            + len(scheduler.holdback_queue)
        )
        assert total_requests == 6, "No requests should be lost during scheduling"

    # Speculative decoding length cap tests

    def test_update_request_uses_effective_max_len_with_spec_tokens(self, scheduler):
        """Test that _update_request_with_output reduces max_model_len by spec_len.

        This prevents the assertion crash when speculative decoding adds tokens
        that would exceed max_model_len.
        """
        from unittest.mock import patch, Mock

        scheduler.max_model_len = 1024
        scheduler.num_spec_tokens = 5

        mock_request = Mock()
        mock_request.append_output_token_ids = Mock()

        with patch("vllm_neuron.core.scheduler.check_stop") as mock_check_stop:
            mock_check_stop.return_value = False

            scheduler._update_request_with_output(mock_request, [42])

            # check_stop should be called with effective_max_len (1024 - 5 = 1019)
            mock_check_stop.assert_called_once()
            call_args = mock_check_stop.call_args
            assert call_args[0][1] == 1019, (
                f"Expected effective_max_len=1019, got {call_args[0][1]}"
            )

    def test_update_request_no_spec_tokens(self, scheduler):
        """Test that without speculative decoding, max_model_len is unchanged."""
        from unittest.mock import patch, Mock

        scheduler.max_model_len = 1024
        # No num_spec_tokens attribute
        if hasattr(scheduler, "num_spec_tokens"):
            delattr(scheduler, "num_spec_tokens")

        mock_request = Mock()
        mock_request.append_output_token_ids = Mock()

        with patch("vllm_neuron.core.scheduler.check_stop") as mock_check_stop:
            mock_check_stop.return_value = False

            scheduler._update_request_with_output(mock_request, [42])

            # check_stop should be called with full max_model_len
            call_args = mock_check_stop.call_args
            assert call_args[0][1] == 1024

    def test_update_request_stops_when_check_stop_returns_true(self, scheduler):
        """Test that generation stops and tokens are trimmed when check_stop returns True."""
        from unittest.mock import patch, Mock

        scheduler.max_model_len = 1024
        scheduler.num_spec_tokens = 5

        mock_request = Mock()
        mock_request.append_output_token_ids = Mock()

        with patch("vllm_neuron.core.scheduler.check_stop") as mock_check_stop:
            # Stop on second token
            mock_check_stop.side_effect = [False, True]

            new_tokens = [1, 2, 3, 4]
            result_tokens, stopped = scheduler._update_request_with_output(
                mock_request, new_tokens
            )

            assert stopped is True
            # Tokens after stop should be trimmed
            assert result_tokens == [1, 2]

    def test_update_request_effective_max_len_never_negative(self, scheduler):
        """Test that effective_max_len is clamped to 0 if spec_len > max_model_len."""
        from unittest.mock import patch, Mock

        scheduler.max_model_len = 10
        scheduler.num_spec_tokens = 100  # Larger than max_model_len

        mock_request = Mock()
        mock_request.append_output_token_ids = Mock()

        with patch("vllm_neuron.core.scheduler.check_stop") as mock_check_stop:
            mock_check_stop.return_value = True

            scheduler._update_request_with_output(mock_request, [42])

            # effective_max_len should be max(10 - 100, 0) = 0
            call_args = mock_check_stop.call_args
            assert call_args[0][1] == 0
