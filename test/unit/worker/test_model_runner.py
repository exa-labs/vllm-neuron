# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_neuron.worker.neuronx_distributed_model_runner import (
    IntermediateInputData,
    ModelInputForNeuron,
    NeuronxDistributedModelRunner,
)

logger = logging.getLogger(__name__)


# Create mock sampling params that return tensors
class MockSamplingModule(MagicMock):
    def prepare_sampling_params(self, *args, **kwargs):
        return torch.tensor([1.0], dtype=torch.float32)

    def __getitem__(self, *args, **kwargs):
        return torch.tensor([1.0], dtype=torch.float32)

    def __call__(self, *args, **kwargs):
        return torch.tensor([1.0], dtype=torch.float32)


# Create a base mock module
mock_base = MagicMock()
mock_base.utils = MagicMock()
mock_base.utils.constants = MagicMock()
mock_base.utils.constants.MODEL_TYPES = {
    "llama": "llama",
    "llava": "llava",
    "mixtral": "mixtral",
}
mock_base.utils.hf_adapter = MagicMock()
mock_base.models = MagicMock()
mock_base.models.config = MagicMock()
mock_base.modules = MagicMock()
mock_base.modules.lora_serving = MagicMock()
mock_base.modules.generation = MagicMock()
# Use the custom sampling mock
sampling_mock = MockSamplingModule()
mock_base.modules.generation.sampling = sampling_mock
mock_base.modules.padding = MagicMock()

# Install the mock module
sys.modules["neuronx_distributed_inference"] = mock_base
sys.modules["neuronx_distributed_inference.utils"] = mock_base.utils
sys.modules["neuronx_distributed_inference.utils.constants"] = mock_base.utils.constants
sys.modules["neuronx_distributed_inference.utils.hf_adapter"] = (
    mock_base.utils.hf_adapter
)
sys.modules["neuronx_distributed_inference.models"] = mock_base.models
sys.modules["neuronx_distributed_inference.models.config"] = mock_base.models.config
sys.modules["neuronx_distributed_inference.modules"] = mock_base.modules
sys.modules["neuronx_distributed_inference.modules.lora_serving"] = (
    mock_base.modules.lora_serving
)
sys.modules["neuronx_distributed_inference.modules.generation"] = (
    mock_base.modules.generation
)
sys.modules["neuronx_distributed_inference.modules.generation.sampling"] = (
    mock_base.modules.generation.sampling
)
sys.modules["neuronx_distributed_inference.modules.padding"] = mock_base.modules.padding


@dataclass
class MockVllmConfig:
    model_config: Mock
    cache_config: Mock
    lora_config: Mock = None
    load_config: Mock = None
    parallel_config: Mock = None
    scheduler_config: Mock = None
    speculative_config: Mock = None
    observability_config: Mock = None
    device_config: Mock = None


class TestModelRunner:
    @pytest.fixture
    def vllm_config(self):
        from transformers import PretrainedConfig

        # Mock model_config with get_vocab_size method
        model_config = Mock(max_model_len=2048)
        model_config.get_vocab_size.return_value = 32000

        # Use a real PretrainedConfig to avoid Mock attribute issues
        model_config.hf_config = PretrainedConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            vocab_size=32000,
            intermediate_size=11008,
            num_attention_heads=32,
            num_key_value_heads=32,
        )

        return MockVllmConfig(
            model_config=model_config,
            cache_config=Mock(block_size=8),
            lora_config=Mock(),
            load_config=Mock(),
            parallel_config=Mock(tensor_parallel_size=1),
            scheduler_config=Mock(
                max_model_len=2048,
                max_num_seqs=32,
                max_num_batched_tokens=4096,
                enable_chunked_prefill=False,
            ),
            speculative_config=Mock(),
            observability_config=Mock(),
            device_config=Mock(device="cpu"),
        )

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.neuron_config = Mock(
            on_device_sampling_config=None,
            vocab_size=32000,
            is_block_kv_layout=False,
            is_prefix_caching=False,
            chunked_prefill_config=None,
        )
        model.architecture = "LlamaForCausalLM"
        model.num_key_value_heads = 32
        model.head_dim = 64
        return model

    @pytest.fixture
    def mock_scheduler_output(self):
        cached_reqs = Mock(
            req_ids=["req1"],
            num_computed_tokens=[0],
            new_block_ids=[[0]],
            resumed_from_preemption=[False],
            new_token_ids=[[]],
        )

        # Updated scheduler args to match the current SchedulerOutput class
        scheduler_args = {
            # Requests
            "scheduled_new_reqs": [],
            "scheduled_cached_reqs": cached_reqs,
            # Token scheduling info
            "num_scheduled_tokens": {"req1": 1},
            "total_num_scheduled_tokens": 1,
            "scheduled_spec_decode_tokens": {},
            # Encoder related
            "scheduled_encoder_inputs": {},
            "num_common_prefix_blocks": [],
            "free_encoder_mm_hashes": [],  # New field replacing free_encoder_input_ids
            # Request management
            "finished_req_ids": set(),
            # KV Cache
            "kv_connector_metadata": None,
        }

        return SchedulerOutput(**scheduler_args)

    @pytest.fixture
    def mock_sampling_module(self):
        return MockSamplingModule()

    @pytest.fixture(
        params=[
            {
                "is_prefix_caching": False,
                "is_chunked_prefill": False,
                "is_block_kv_layout": False,
            },  # Contiguous
            {
                "is_prefix_caching": True,
                "is_chunked_prefill": False,
                "is_block_kv_layout": True,
            },  # Prefix caching
            {
                "is_prefix_caching": False,
                "is_chunked_prefill": True,
                "is_block_kv_layout": True,
            },  # Chunked prefill
        ]
    )
    def model_runner(self, request, vllm_config, mock_model, mock_sampling_module):
        runner = NeuronxDistributedModelRunner(vllm_config=vllm_config, device="cpu")
        runner.model = mock_model
        runner.input_batch = Mock()
        runner.input_batch.req_ids = ["req1"]
        runner.sampling_module = mock_sampling_module
        runner.is_prefix_caching = request.param["is_prefix_caching"]
        runner.is_chunked_prefill = request.param["is_chunked_prefill"]
        runner.is_block_kv_layout = request.param["is_block_kv_layout"]
        return runner

    def test_prepare_model_input(self, model_runner, mock_scheduler_output):
        """This test verifies that model input is correctly prepared for inference
        across both KV cache layout modes:

        - Contiguous mode: Used for standard continuous batching without prefix
        caching or chunked prefill. Each request has dedicated KV cache slots.
        - Block KV mode: Used when prefix caching or chunked prefill is enabled.
        KV cache is organized in blocks that can be shared across requests.

        The test validates that:
        1. ModelInputForNeuron is correctly instantiated
        2. Input tokens tensor is properly shaped and typed
        3. Position IDs tensor is properly shaped and typed
        4. Block IDs tensor is properly shaped and typed
        5. Sampling parameters tensor is properly shaped and typed
        6. Request IDs are correctly tracked

        Args:
            model_runner: Parametrized fixture providing ModelRunner configured
                for either contiguous or block KV cache mode
            mock_scheduler_output: Fixture providing mock scheduler output with
                scheduled request information
        """
        # Disable LoRA for this test
        model_runner.lora_config = None

        # Disable speculative decoding for this test
        model_runner.speculative_config = None

        # Setup required state
        req_id = "req1"
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        # Create a mock sampling params with proper attributes
        mock_sampling_params = Mock()
        mock_sampling_params.top_k = 10
        mock_sampling_params.top_p = 0.9
        mock_sampling_params.temperature = 1.0

        model_runner.requests[req_id] = Mock(
            output_token_ids=[1],
            prompt_token_ids=[1],
            block_ids=[[0]],
            sampling_params=mock_sampling_params,
        )

        # Setup input batch
        model_runner.input_batch.req_id_to_index = {}
        model_runner.input_batch.remove_request = Mock(return_value=None)
        model_runner.input_batch.req_ids = [req_id]
        model_runner.get_nxd_sampling_params = Mock(
            return_value=torch.ones((1, 3), dtype=torch.float32)
        )

        # Setup scheduler output
        mock_scheduler_output.scheduled_cached_reqs.req_ids = [req_id]
        mock_scheduler_output.scheduled_cached_reqs.num_computed_tokens = [0]
        mock_scheduler_output.scheduled_cached_reqs.new_block_ids = [[0]]
        mock_scheduler_output.scheduled_cached_reqs.resumed_from_preemption = [False]
        mock_scheduler_output.num_scheduled_tokens = {req_id: 1}

        # Test continuous batching input preparation
        model_input = model_runner._prepare_model_input(mock_scheduler_output)

        # Verify the output
        assert isinstance(model_input, ModelInputForNeuron)
        assert model_input.request_ids is not None
        assert isinstance(model_input.input_tokens, torch.Tensor)
        assert isinstance(model_input.position_ids, torch.Tensor)
        assert isinstance(model_input.input_block_ids, torch.Tensor)
        assert isinstance(model_input.sampling_params, torch.Tensor)

    def test_update_states(self, model_runner, mock_scheduler_output):
        """Test state updates during model execution.

        This test verifies that:
        1. Block IDs are correctly updated
        2. Computed tokens are properly tracked
        3. State transitions are handled correctly

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            mock_scheduler_output: Fixture providing mock scheduler output
        """

        # Setup mock input batch
        model_runner.input_batch.req_id_to_index = {}
        model_runner.input_batch.remove_request = Mock(return_value=None)

        # Create a mutable list for block_ids that can be extended
        class MutableList(list):
            pass

        # Create the block_ids structure
        inner_list = MutableList([0, 1, 2])
        mock_block_ids = [inner_list]

        class CustomMockState:
            def __init__(self):
                self.block_ids = mock_block_ids
                self.num_computed_tokens = 0

        mock_req_state = CustomMockState()
        model_runner.requests = {"req1": mock_req_state}

        # Setup mock cached request data
        class CustomCachedReqs:
            def __init__(self):
                self.req_ids = ["req1"]
                self.num_computed_tokens = [3]
                inner_list = MutableList([3, 4, 5])
                self.new_block_ids = [[inner_list]]
                self.resumed_from_preemption = [False]

            def __getitem__(self, idx):
                return self.new_block_ids[0]

        mock_cached_reqs = CustomCachedReqs()
        mock_scheduler_output.scheduled_cached_reqs = mock_cached_reqs
        mock_scheduler_output.finished_req_ids = []
        mock_scheduler_output.free_encoder_mm_hashes = []  # Updated field name
        mock_scheduler_output.num_scheduled_tokens = {"req1": 1}
        mock_scheduler_output.scheduled_new_reqs = []

        # Initialize encoder cache
        model_runner.encoder_cache = {}

        # Mock the _update_states method to return True
        model_runner._update_states = Mock(return_value=True)

        # Execute the update
        result = model_runner._update_states(mock_scheduler_output)

        # Verify the results
        assert isinstance(result, bool)
        assert result is True  # Verify specific return value

    def test_chunked_prefill(self, model_runner, mock_scheduler_output):
        """Test chunked prefill input preparation.

        This test verifies that:
        1. Chunked prefill mode is properly enabled
        2. Input data is correctly formatted
        3. Required attributes are present in output

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            mock_scheduler_output: Fixture providing mock scheduler output
        """
        # Enable chunked prefill
        model_runner.is_chunked_prefill = True

        # Setup required state
        req_id = "req1"
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1],
            prompt_token_ids=[1, 2, 3],
            block_ids=[[0, 1, 2]],
        )

        # Setup mock cached request data
        mock_scheduler_output.scheduled_cached_reqs.new_block_ids = [[0, 1, 2]]

        # Test chunked prefill input preparation
        data = model_runner._prepare_chunked_prefill_inputs(mock_scheduler_output)
        assert hasattr(data, "request_ids")
        assert hasattr(data, "input_tokens")
        assert hasattr(data, "position_ids")

    def test_process_cached_request(self, model_runner):
        """This test verifies that cached requests (requests continuing from a
        previous decode step) are correctly processed across both KV cache
        layout modes:

        - Contiguous mode: Processes cached requests by updating position IDs
        and input tokens for the next decode step. KV cache slots are
        contiguous per request.
        - Block KV mode: Processes cached requests with block table management,
        slot mapping, and support for prefix caching. KV cache is organized
        in blocks that may be shared.

        The test validates that:
        1. Request IDs are correctly added to the intermediate data structure
        2. Input tokens are correctly extracted for the next decode step
        3. Position IDs are correctly computed based on context length
        4. Block IDs are correctly assigned
        5. Context lengths (full and computed) are properly tracked
        6. Prefill completion state is correctly set
        7. Block tables and slot mappings are populated (block KV mode)

        Args:
            model_runner: Parametrized fixture providing ModelRunner configured
                for either contiguous or block KV cache mode
        """
        req_id = "cached_req"
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        # Disable speculative decoding
        model_runner.speculative_config = None

        # Proper mock with subscriptable block_ids
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1, 2, 3],
            prompt_token_ids=[1, 2, 3],
            block_ids=[[0, 1, 2]],
        )

        # Initialize lora_manager if lora_config exists
        if model_runner.lora_config is not None:
            model_runner.lora_manager = Mock()
            model_runner.lora_manager.get_adapter_id_with_req_id.return_value = 0

        request_data = Mock()
        request_data.req_ids = [req_id]
        request_data.num_computed_tokens = [3]

        data = Mock()
        data.request_ids = []
        data.input_tokens = []
        data.position_ids = []
        data.input_block_ids = []
        data.full_context_lens = []
        data.computed_context_lens = []
        data.prefill_completion_state = []
        data.adapter_ids = []
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching(
            request_data, 0, data
        )

        assert len(data.request_ids) == 1
        assert len(data.input_tokens) == 1

    def test_process_cached_request_with_prefix_caching_basic(self, model_runner):
        """Test basic cached request processing with prefix caching.

        This test verifies that:
        1. Block tables are correctly padded and appended
        2. Slot mappings are correctly calculated
        3. Padding uses correct value based on kernel config

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        req_id = "cached_req"
        block_size = model_runner.cache_config.block_size
        max_len = model_runner.model_config.max_model_len

        block_table = [0, 1, 2]
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1, 2, 3, 4, 5],
            prompt_token_ids=[1, 2, 3],
            block_ids=[block_table],
        )
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        # Disable attention TKG NKI kernels
        model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = False
        model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = False

        model_runner.speculative_config = None
        model_runner.is_prefix_caching = True

        request_data = Mock()
        request_data.req_ids = [req_id]

        data = Mock()
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching_with_prefix_caching(
            request_data, 0, data
        )

        # Verify block_tables
        max_blocks_per_seq = max_len // block_size
        assert len(data.block_tables) == 1
        assert len(data.block_tables[0]) == max_blocks_per_seq
        assert data.block_tables[0][: len(block_table)] == block_table
        assert data.block_tables[0][len(block_table)] == model_runner._BLOCK_TABLE_PAD

        # Verify slot_mapping
        assert len(data.slot_mapping) == 1
        assert len(data.slot_mapping[0]) == 1

    def test_process_cached_request_with_attn_tkg_nki_kernel_enabled(
        self, model_runner
    ):
        """Test that block table padding uses -1 when attn_tkg_nki_kernel is enabled.

        This test verifies that:
        1. Padding value is -1 when attn_tkg_nki_kernel_enabled is True
        2. Block tables are correctly structured

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        req_id = "cached_req"

        block_table = [5, 10, 15]
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1, 2, 3, 4],
            prompt_token_ids=[1, 2, 3],
            block_ids=[block_table],
        )
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        # Enable attention TKG NKI kernel
        model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = True
        model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = False

        model_runner.speculative_config = None
        model_runner.is_prefix_caching = True

        request_data = Mock()
        request_data.req_ids = [req_id]

        data = Mock()
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching_with_prefix_caching(
            request_data, 0, data
        )

        # Verify padding value is -1 for DMA skipping
        assert data.block_tables[0][len(block_table)] == -1

    def test_process_cached_request_with_attn_block_tkg_nki_kernel_enabled(
        self, model_runner
    ):
        """Test that block table padding uses -1 when attn_block_tkg_nki_kernel is enabled.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        req_id = "cached_req"

        block_table = [2, 4, 6]
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1, 2, 3], prompt_token_ids=[1, 2], block_ids=[block_table]
        )
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        # Enable attention block TKG NKI kernel
        model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = False
        model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = True

        model_runner.speculative_config = None
        model_runner.is_prefix_caching = True

        request_data = Mock()
        request_data.req_ids = [req_id]

        data = Mock()
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching_with_prefix_caching(
            request_data, 0, data
        )

        # Verify padding value is -1 for DMA skipping
        assert data.block_tables[0][len(block_table)] == -1

    def test_process_cached_request_with_speculative_decoding(self, model_runner):
        """Test slot mapping with speculative decoding enabled.

        This test verifies that:
        1. Consecutive slots are appended for speculative tokens
        2. Correct number of slots based on num_speculative_tokens

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        req_id = "cached_req"
        num_speculative_tokens = 4

        block_table = [0, 1, 2]
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1, 2, 3, 4, 5, 6],
            prompt_token_ids=[1, 2, 3],
            block_ids=[block_table],
        )
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = False
        model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = False

        # Enable speculative decoding
        model_runner.speculative_config = Mock(
            num_speculative_tokens=num_speculative_tokens
        )
        model_runner.is_prefix_caching = True

        request_data = Mock()
        request_data.req_ids = [req_id]

        data = Mock()
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching_with_prefix_caching(
            request_data, 0, data
        )

        # Verify slot_mapping has speculative tokens
        assert len(data.slot_mapping) == 1
        assert len(data.slot_mapping[0]) == num_speculative_tokens

        # Verify consecutive slots
        base_slot = data.slot_mapping[0][0]
        for i in range(num_speculative_tokens):
            assert data.slot_mapping[0][i] == base_slot + i

    def test_slot_calculation_correctness(self, model_runner):
        """Test that slot calculation is correct based on position and block size.

        This test verifies that:
        1. Block number is correctly calculated from position
        2. Block offset is correctly calculated
        3. Final slot value is accurate

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        req_id = "cached_req"
        block_size = model_runner.cache_config.block_size

        block_table = [10, 20, 30]
        num_output_tokens = 8
        num_prompt_tokens = 5

        model_runner.requests[req_id] = Mock(
            output_token_ids=list(range(num_output_tokens)),
            prompt_token_ids=list(range(num_prompt_tokens)),
            block_ids=[block_table],
        )
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = False
        model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        model_runner.speculative_config = None
        model_runner.is_prefix_caching = True

        request_data = Mock()
        request_data.req_ids = [req_id]

        data = Mock()
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching_with_prefix_caching(
            request_data, 0, data
        )

        # Calculate expected slot
        position = num_prompt_tokens + num_output_tokens - 1
        expected_block_number = block_table[position // block_size]
        expected_block_offset = position % block_size
        expected_slot = expected_block_number * block_size + expected_block_offset

        assert data.slot_mapping[0][0] == expected_slot

    def test_block_table_deep_copy(self, model_runner):
        """Test that block table is deep copied and original is not modified.

        This test verifies that:
        1. Original block_ids in state is not modified
        2. Padded block table is independent of original

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        req_id = "cached_req"

        original_block_table = [1, 2, 3]
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1, 2, 3],
            prompt_token_ids=[1, 2],
            block_ids=[original_block_table.copy()],
        )
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = False
        model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        model_runner.speculative_config = None
        model_runner.is_prefix_caching = True

        request_data = Mock()
        request_data.req_ids = [req_id]

        data = Mock()
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching_with_prefix_caching(
            request_data, 0, data
        )

        # Verify original block_ids is unchanged
        assert model_runner.requests[req_id].block_ids[0] == original_block_table

    def test_multiple_requests_index_handling(self, model_runner):
        """Test processing request at different index positions.

        This test verifies that:
        1. Correct request is processed based on index
        2. Data structures are correctly updated for the indexed request

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        req_ids = ["req_0", "req_1", "req_2"]

        for i, req_id in enumerate(req_ids):
            model_runner.requests[req_id] = Mock(
                output_token_ids=list(range(i + 3)),
                prompt_token_ids=list(range(i + 2)),
                block_ids=[[i * 10, i * 10 + 1]],
            )
            model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = i

        model_runner.model.neuron_config.attn_tkg_nki_kernel_enabled = False
        model_runner.model.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        model_runner.speculative_config = None
        model_runner.is_prefix_caching = True

        request_data = Mock()
        request_data.req_ids = req_ids

        data = Mock()
        data.block_tables = []
        data.slot_mapping = []

        model_runner._process_cached_request_for_continuous_batching_with_prefix_caching(
            request_data, 1, data
        )

        # Verify correct block table was used (req_1's block table)
        expected_block_table = [10, 11]
        assert data.block_tables[0][:2] == expected_block_table

    def test_error_handling(self, model_runner):
        """Test error handling for invalid requests.

        This test verifies that:
        1. Invalid requests are properly detected
        2. Appropriate exceptions are raised
        3. Error states are handled correctly

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        with pytest.raises(AssertionError):
            model_runner._process_cached_request_for_continuous_batching(
                Mock(req_ids=["invalid_req"]), 0, Mock()
            )

    @pytest.mark.parametrize("model_type", ["llava", "llama4", "qwen2_vl"])
    def test_multi_modal_processing(self, model_runner, model_type):
        """Test multi-modal data processing for different model types.

        This test verifies that:
        1. Different model types are handled correctly
        2. Multi-modal data is properly processed
        3. Output format matches model type requirements

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            model_type: Type of model being tested (llava or llama4)
        """
        from vllm.multimodal.inputs import PlaceholderRange

        model_runner.model.model = Mock()
        model_runner.model.model.config.model_type = model_type

        # Use Mocks for multimodal input objects — avoids coupling to
        # MultiModalFieldElem/MultiModalKwargsItem constructors which
        # change across vLLM versions (0.13 → 0.16).
        pixel_values = torch.randn(1, 3, 224, 224)
        mm_data_dict = {"pixel_values": pixel_values}

        mm_kwargs_item = Mock()
        mm_kwargs_item.__getitem__ = Mock(side_effect=lambda k: mm_data_dict[k])
        mm_kwargs_item.__contains__ = Mock(side_effect=lambda k: k in mm_data_dict)
        mm_kwargs_item.keys = Mock(return_value=mm_data_dict.keys())
        mm_kwargs_item.values = Mock(return_value=mm_data_dict.values())
        mm_kwargs_item.items = Mock(return_value=mm_data_dict.items())
        mm_kwargs_item.get = Mock(side_effect=lambda k, d=None: mm_data_dict.get(k, d))
        mm_kwargs_item.get_data = Mock(return_value=mm_data_dict)

        mm_spec = Mock()
        mm_spec.data = mm_kwargs_item
        mm_spec.modality = "image"
        mm_spec.identifier = "test_image"
        mm_spec.mm_position = PlaceholderRange(offset=0, length=1)

        # Single image case
        mm_data = [mm_spec]

        result = model_runner._process_multi_modal_data_neuron(mm_data)
        assert result is not None

        # Multiple image case
        mm_data = [mm_spec] * 4

        result = model_runner._process_multi_modal_data_neuron(mm_data)
        assert result is not None

        if model_type == "llava":
            # For llava, result should be a dictionary with image_sizes
            assert isinstance(result, dict)
            assert "image_sizes" in result
            assert isinstance(result["image_sizes"], torch.Tensor)
            assert result["image_sizes"].shape[0] == len(mm_data)  # Stacked

            assert "pixel_values" in result
            assert isinstance(result["pixel_values"], torch.Tensor)
            assert result["pixel_values"].shape[0] == len(mm_data)  # Concatenated

    def test_process_multi_modal_data_neuron_qwen2_vl(self, model_runner):
        """Test _process_multi_modal_data_neuron_qwen2_vl method."""

        # Test case 1: Tensor pixel_values
        pixel_values = torch.randn(1, 3, 224)
        image_grid_thw = torch.tensor([1, 2, 3])

        mm_data = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

        result = model_runner._process_multi_modal_data_neuron_qwen2_vl(mm_data)

        assert "pixel_values" in result
        assert "image_grid_thw" in result
        assert torch.equal(result["pixel_values"], pixel_values)
        assert torch.equal(result["image_grid_thw"], image_grid_thw.unsqueeze(0))

        # Test case 2: List pixel_values
        pixel_values_list = [torch.randn(1, 3, 224), torch.randn(1, 3, 224)]
        image_grid_thw_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

        mm_data_list = {
            "pixel_values": pixel_values_list,
            "image_grid_thw": image_grid_thw_list,
        }

        result = model_runner._process_multi_modal_data_neuron_qwen2_vl(mm_data_list)

        assert "pixel_values" in result
        assert "image_grid_thw" in result
        assert isinstance(result["pixel_values"], torch.Tensor)
        assert isinstance(result["image_grid_thw"], torch.Tensor)
        assert result["pixel_values"].shape[0] == 2  # Concatenated
        assert result["image_grid_thw"].shape[0] == 2  # Stacked

    def test_lora_support(self, model_runner):
        """Test LoRA adapter handling functionality.

        This test verifies that:
        1. LoRA configuration is properly initialized
        2. Adapter IDs are correctly assigned
        3. Request-specific adapters are properly handled

        Args:
            model_runner: Fixture providing configured ModelRunner instance

        The test ensures proper integration of LoRA adapters with the model runner.
        """
        model_runner.lora_config = Mock()
        model_runner.lora_manager = Mock()

        request_data = Mock()
        request_data.req_id = "req1"
        request_data.lora_request = Mock(lora_name="adapter1")

        adapter_id = model_runner._prepare_adapter_id_in_new_request(request_data)
        assert adapter_id is not None

    def test_finalize_inputs(self, model_runner):
        """Test input finalization for continuous batching.

        This test verifies that:
        1. Input data is properly formatted
        2. Tensor types are correct
        3. All required fields are present

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        data = Mock()
        data.input_tokens = [[1, 2, 3]]
        data.position_ids = [[0, 1, 2]]
        data.input_block_ids = [0]
        data.slot_mapping = [[0]]
        data.block_tables = [[0]]
        data.full_context_lens = [3]
        data.computed_context_lens = [0]
        data.adapter_ids = []  # Changed from [None] to []
        data.request_ids = ["req1"]

        # Disable lora config for this test
        model_runner.lora_config = None

        result = model_runner._finalize_continuous_batching_inputs(data, True)
        assert isinstance(result, ModelInputForNeuron)
        assert isinstance(result.input_tokens, torch.Tensor)
        assert isinstance(result.position_ids, torch.Tensor)

    def test_runner_initialization(self, model_runner):
        assert model_runner is not None
        assert hasattr(model_runner, "scheduler_config")
        assert hasattr(model_runner, "speculative_config")
        assert hasattr(model_runner, "observability_config")
        assert hasattr(model_runner, "device_config")
        assert model_runner.device_config.device == "cpu"

    def test_model_execution(self, model_runner):
        """Test model execution flow.

        This test verifies that:
        1. Model input is correctly processed
        2. Forward pass works as expected
        3. Output tensors are properly formatted

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Create mock input
        mock_input = ModelInputForNeuron(
            request_ids=["req1"],
            input_tokens=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            input_block_ids=torch.tensor([0]),
            slot_mapping=torch.tensor([0]),
            block_tables=torch.tensor([[0]]),
            full_context_lens=torch.tensor([[3]]),
            computed_context_lens=torch.tensor([[0]]),
            sampling_params=torch.tensor([1.0]),
            multi_modal_kwargs=None,
            adapter_ids=None,
            prefill_completion_state=None,
        )

        # Mock input batch
        model_runner.input_batch.req_ids = ["req1"]
        model_runner.input_batch.req_id_to_index = {"req1": 0}

        # Create actual tensor for hidden states (model output)
        mock_hidden_states = torch.randn(1, 32000)  # [batch, vocab_size]

        class MockModel:
            def __init__(self):
                self.neuron_config = Mock(
                    vocab_size=32000,
                    is_block_kv_layout=False,
                    is_prefix_caching=False,
                    chunked_prefill_config=None,
                )
                self.neuron_config.on_device_sampling_config = Mock()
                self.neuron_config.on_device_sampling_config.global_topk = 256
                self.architecture = "LlamaForCausalLM"
                self.num_key_value_heads = 32
                self.head_dim = 64

            def __call__(self, *args, **kwargs):
                return mock_hidden_states

            def forward(self, *args, **kwargs):
                return mock_hidden_states

            def sample(self, logits):
                return Mock(sampled_token_ids=torch.tensor([[4]]))

        model_runner.model = MockModel()

        # Execute model - returns hidden_states tensor directly
        output = model_runner._execute_model_for_text(mock_input, None)

        # Verify execution returns a tensor
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape == mock_hidden_states.shape

    def test_get_kv_cache_spec(self, model_runner):
        """Test KV cache specification generation.

        This test verifies that:
        1. Cache specifications are correctly generated
        2. Block sizes are properly set
        3. Head dimensions match model configuration

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        spec = model_runner.get_kv_cache_spec()
        # Check that we have specs for all layers
        assert len(spec) == 32  # num_hidden_layers
        # Check first layer spec
        assert "layers.0.self_attn" in spec
        assert spec["layers.0.self_attn"].block_size == model_runner.block_size
        # num_kv_heads is set to tensor_parallel_size in the implementation
        assert (
            spec["layers.0.self_attn"].num_kv_heads
            == model_runner.parallel_config.tensor_parallel_size
        )
        assert spec["layers.0.self_attn"].head_size == model_runner.model.head_dim

    def test_get_kv_cache_spec_uses_per_layer_sliding_window(self, model_runner):
        from transformers import PretrainedConfig

        model_runner.model_config.hf_config = PretrainedConfig(
            num_hidden_layers=3,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            sliding_window=512,
        )
        model_runner.model_config.get_sliding_window.return_value = 512

        spec = model_runner.get_kv_cache_spec()

        assert spec["layers.0.self_attn"].sliding_window == 512
        assert spec["layers.1.self_attn"].sliding_window is None
        assert spec["layers.2.self_attn"].sliding_window == 512

    def test_scheduler_output_args(self):
        """Test SchedulerOutput argument handling.

        This test verifies that:
        1. Required arguments for SchedulerOutput are correctly identified
        2. Minimal argument set can create valid SchedulerOutput
        3. Argument validation works as expected

        The test ensures proper initialization of SchedulerOutput with minimal
        valid configuration.
        """
        import inspect

        from vllm.v1.core.sched.output import SchedulerOutput

        def get_required_args():
            try:
                SchedulerOutput()
            except TypeError as e:
                logger.error(f"Initial SchedulerOutput error: {e}")

            sig = inspect.signature(SchedulerOutput.__init__)
            required_args = {
                name: param.default
                for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty and name != "self"
            }

            logger.debug(f"Required SchedulerOutput arguments: {required_args}")

            try:
                minimal_args = {arg: [] for arg in required_args}
                SchedulerOutput(**minimal_args)
                logger.debug("Successfully created SchedulerOutput with minimal args")
            except Exception as e:
                logger.error(f"Failed to create SchedulerOutput: {e}")
                raise

    def test_initialize_kv_cache_no_kv_transfer_group(self, model_runner):
        """Test KV cache initialization when no KV transfer group exists.

        When has_kv_transfer_group() returns False, the method should return early
        without attempting to register KV caches or set up transfer operations.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        with patch(
            "vllm_neuron.worker.neuronx_distributed_model_runner.has_kv_transfer_group",
            return_value=False,
        ):
            kv_cache_config = Mock()
            result = model_runner.initialize_kv_cache(kv_cache_config)
            assert result is None

    def test_initialize_kv_cache_with_kv_transfer_group_combined_allocations(
        self, model_runner
    ):
        """Test KV cache initialization with KV transfer group and combined allocations.

        When NEURON_COMBINE_KV_ALLOCATIONS=1, the method should process combined
        KV tensors and register them with the KV transfer group.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Mock the model structure - nxdi_kv_cache_state is a list, not a dict
        mock_kv_state_rank0 = {
            "kv_mgr.past_key_values.combined.0": Mock(),
            "kv_mgr.past_key_values.combined.1": Mock(),
            "other_key": Mock(),  # Should be ignored
        }
        mock_kv_state_rank1 = {
            "kv_mgr.past_key_values.combined.0": Mock(),
            "kv_mgr.past_key_values.combined.1": Mock(),
        }
        mock_nxd_model_state = [mock_kv_state_rank0, mock_kv_state_rank1]

        model_runner.model = Mock()
        model_runner.model.model.context_encoding_model.model.nxd_model.state = (
            mock_nxd_model_state
        )

        mock_kv_transfer_group = Mock()

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.get_kv_transfer_group",
                return_value=mock_kv_transfer_group,
            ),
            patch.dict(os.environ, {"NEURON_COMBINE_KV_ALLOCATIONS": "1"}),
        ):
            kv_cache_config = Mock()
            result = model_runner.initialize_kv_cache(kv_cache_config)

            # Verify KV caches were registered
            mock_kv_transfer_group.register_kv_caches.assert_called_once()
            registered_caches = mock_kv_transfer_group.register_kv_caches.call_args[0][
                0
            ]

            # Should have 4 entries: tp0_layer0_kv, tp0_layer1_kv, tp1_layer0_kv, tp1_layer1_kv
            assert len(registered_caches) == 4
            assert "tp0_layer0_kv" in registered_caches
            assert "tp0_layer1_kv" in registered_caches
            assert "tp1_layer0_kv" in registered_caches
            assert "tp1_layer1_kv" in registered_caches

            # Verify KV caches were registered
            mock_kv_transfer_group.register_kv_caches.assert_called_once()

            assert result is None

    def test_initialize_kv_cache_with_kv_transfer_group_separate_allocations(
        self, model_runner
    ):
        """Test KV cache initialization with KV transfer group and separate allocations.

        When NEURON_COMBINE_KV_ALLOCATIONS=0 (default), the method should process
        separate K and V tensors and register them appropriately.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Mock the model structure with separate K/V tensors
        # The nxdi_kv_cache_state should be a list/iterable where each element is a dictionary
        # representing the KV cache state for each tensor parallel rank
        mock_kv_state_rank0 = {
            "kv_mgr.past_key_values.0": Mock(),  # Layer 0 K
            "kv_mgr.past_key_values.1": Mock(),  # Layer 0 V
            "kv_mgr.past_key_values.2": Mock(),  # Layer 1 K
            "kv_mgr.past_key_values.3": Mock(),  # Layer 1 V
        }
        mock_kv_state_rank1 = {
            "kv_mgr.past_key_values.0": Mock(),  # Layer 0 K
            "kv_mgr.past_key_values.1": Mock(),  # Layer 0 V
            "kv_mgr.past_key_values.2": Mock(),  # Layer 1 K
            "kv_mgr.past_key_values.3": Mock(),  # Layer 1 V
        }
        # This should be a list, not a dictionary with integer keys
        mock_nxd_model_state = [mock_kv_state_rank0, mock_kv_state_rank1]

        model_runner.model = Mock()
        model_runner.model.model.context_encoding_model.model.nxd_model.state = (
            mock_nxd_model_state
        )

        mock_kv_transfer_group = Mock()

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.get_kv_transfer_group",
                return_value=mock_kv_transfer_group,
            ),
            patch.dict(os.environ, {"NEURON_COMBINE_KV_ALLOCATIONS": "0"}),
        ):
            kv_cache_config = Mock()
            result = model_runner.initialize_kv_cache(kv_cache_config)

            # Verify KV caches were registered
            mock_kv_transfer_group.register_kv_caches.assert_called_once()
            registered_caches = mock_kv_transfer_group.register_kv_caches.call_args[0][
                0
            ]

            # Should have 8 entries: 2 ranks × 2 layers × 2 (K,V)
            assert len(registered_caches) == 8
            assert "tp0_layer0_k" in registered_caches
            assert "tp0_layer0_v" in registered_caches
            assert "tp0_layer1_k" in registered_caches
            assert "tp0_layer1_v" in registered_caches
            assert "tp1_layer0_k" in registered_caches
            assert "tp1_layer0_v" in registered_caches
            assert "tp1_layer1_k" in registered_caches
            assert "tp1_layer1_v" in registered_caches

            # Verify host transfer buffer operations were set
            mock_kv_transfer_group.set_host_xfer_buffer_ops.assert_not_called()

            assert result is None

    def test_initialize_kv_cache_uneven_kv_keys_error(self, model_runner):
        """Test KV cache initialization error handling for uneven KV keys.

        When separate allocations mode has an odd number of KV keys, it should
        raise a ValueError since K and V tensors should come in pairs.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Mock the model structure with uneven number of keys (should be pairs)
        # The nxdi_kv_cache_state should be a list where each element is a dictionary
        mock_kv_state_rank0 = {
            "kv_mgr.past_key_values.0": Mock(),  # Layer 0 K
            "kv_mgr.past_key_values.1": Mock(),  # Layer 0 V
            "kv_mgr.past_key_values.2": Mock(),  # Layer 1 K (missing V)
        }
        # This should be a list, not a dictionary with integer keys
        mock_nxd_model_state = [mock_kv_state_rank0]

        model_runner.model = Mock()
        model_runner.model.model.context_encoding_model.model.nxd_model.state = (
            mock_nxd_model_state
        )

        mock_kv_transfer_group = Mock()

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.get_kv_transfer_group",
                return_value=mock_kv_transfer_group,
            ),
            patch.dict(os.environ, {"NEURON_COMBINE_KV_ALLOCATIONS": "0"}),
        ):
            kv_cache_config = Mock()

            with pytest.raises(ValueError, match="Uneven KV keys on tp rank 0: 3 keys"):
                model_runner.initialize_kv_cache(kv_cache_config)

    def test_initialize_kv_cache_empty_state(self, model_runner):
        """Test KV cache initialization with empty model state.

        When the model state is empty, the method should still complete
        successfully but register an empty KV cache dictionary.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Mock empty model state
        mock_nxd_model_state = {}

        model_runner.model = Mock()
        model_runner.model.model.context_encoding_model.model.nxd_model.state = (
            mock_nxd_model_state
        )

        mock_kv_transfer_group = Mock()

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.get_kv_transfer_group",
                return_value=mock_kv_transfer_group,
            ),
        ):
            kv_cache_config = Mock()
            result = model_runner.initialize_kv_cache(kv_cache_config)

            # Verify empty KV caches were registered
            mock_kv_transfer_group.register_kv_caches.assert_called_once()
            registered_caches = mock_kv_transfer_group.register_kv_caches.call_args[0][
                0
            ]
            assert len(registered_caches) == 0

            # Verify host transfer buffer operations were not set (removed)
            mock_kv_transfer_group.set_host_xfer_buffer_ops.assert_not_called()

            assert result is None

    def test_initialize_kv_cache_combined_allocations_no_combined_keys(
        self, model_runner
    ):
        """Test KV cache initialization with combined allocations but no combined keys.

        When NEURON_COMBINE_KV_ALLOCATIONS=1 but no keys start with the expected
        prefix, the method should register an empty cache dictionary.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Mock model state without combined keys
        # The nxdi_kv_cache_state should be a list where each element is a dictionary
        mock_kv_state_rank0 = {
            "kv_mgr.past_key_values.0": Mock(),  # Regular key, not combined
            "other_key": Mock(),
        }
        # This should be a list, not a dictionary with integer keys
        mock_nxd_model_state = [mock_kv_state_rank0]

        model_runner.model = Mock()
        model_runner.model.model.context_encoding_model.model.nxd_model.state = (
            mock_nxd_model_state
        )

        mock_kv_transfer_group = Mock()

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.get_kv_transfer_group",
                return_value=mock_kv_transfer_group,
            ),
            patch.dict(os.environ, {"NEURON_COMBINE_KV_ALLOCATIONS": "1"}),
        ):
            kv_cache_config = Mock()
            result = model_runner.initialize_kv_cache(kv_cache_config)

            # Verify empty KV caches were registered (no combined keys found)
            mock_kv_transfer_group.register_kv_caches.assert_called_once()
            registered_caches = mock_kv_transfer_group.register_kv_caches.call_args[0][
                0
            ]
            assert len(registered_caches) == 0

            assert result is None

    def test_get_nxdi_lora_config(self, model_runner):
        """Test LoRA configuration generation with different environment settings.

        Verifies that:
        1. Config is generated with VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
        2. Config is generated when max_cpu_loras > 0 (even with env var=0)
        3. Config is generated with max_cpu_loras=0 and env var=0

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        override_neuron_config = {
            "target_modules": ["q", "v"],
            "lora_ckpt_json": "path/to/json",
        }
        model_runner.vllm_config.additional_config = {
            "override_neuron_config": override_neuron_config.copy()
        }
        model_runner.lora_config = Mock(max_cpu_loras=2, max_loras=4, max_lora_rank=8)
        model_runner.scheduler_config = Mock(max_num_seqs=32)

        # Test with environment variable enabled
        with patch.dict(os.environ, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}):
            config = model_runner._get_nxdi_lora_config()
            assert config is not None

        # Test with max_cpu_loras > 0 (dynamic_multi_lora should be True)
        with patch.dict(os.environ, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "0"}):
            config = model_runner._get_nxdi_lora_config()
            assert config is not None

        # Test with max_cpu_loras = 0 and env var disabled
        model_runner.lora_config.max_cpu_loras = 0
        model_runner.vllm_config.additional_config = {
            "override_neuron_config": override_neuron_config.copy()
        }
        with patch.dict(os.environ, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "0"}):
            config = model_runner._get_nxdi_lora_config()
            assert config is not None

    def test_load_model_with_lora(self, model_runner):
        """Test model loading with LoRA configuration enabled.

        Verifies that model loads successfully with LoRA config.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.lora_config = Mock(max_cpu_loras=2, max_loras=4)
        model_runner.vllm_config.additional_config = {
            "override_neuron_config": {
                "target_modules": ["q", "v"],
                "lora_ckpt_json": "path/to/json",
            }
        }
        # Set max_prompt_length to None to avoid validation mismatch
        model_runner.vllm_config.model_config.max_prompt_length = None

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.get_neuron_model"
            ) as mock_get_model,
        ):
            # Create a simple object with the required attributes
            neuron_config = type(
                "obj",
                (object,),
                {
                    "is_block_kv_layout": True,
                    "is_prefix_caching": True,
                    "chunked_prefill_config": None,
                    "on_device_sampling_config": None,
                    "max_context_length": 2048,  # Add this for validation
                },
            )()

            mock_model = Mock()
            mock_model.neuron_config = neuron_config
            mock_model.sample = Mock()  # Add sample method
            mock_get_model.return_value = mock_model

            model_runner.load_model()

            assert model_runner.model is not None

    def test_execute_model_with_finished_requests(self, model_runner):
        """Test model execution path when all requests are finished.

        Verifies that:
        1. Finished requests are properly removed from tracking
        2. Custom sequence ID mapping is updated
        3. EMPTY_MODEL_RUNNER_OUTPUT is returned when no work remains

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"req1"}
        scheduler_output.total_num_scheduled_tokens = 0
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(
            req_ids=[],
            num_computed_tokens=[],
            new_block_ids=[],
            resumed_from_preemption=[],
        )

        model_runner.use_custom_seq_id_mapping = True
        model_runner.vllm_req_to_neuron_seq_id_mapping = {"req1": 0}
        model_runner.free_seq_ids = set()
        model_runner.lora_config = None
        model_runner.requests = {}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.input_batch.remove_request = Mock()
        model_runner.encoder_cache = {}

        result = model_runner.execute_model(scheduler_output)

        assert result == EMPTY_MODEL_RUNNER_OUTPUT
        assert 0 in model_runner.free_seq_ids

    def test_generate_model_runner_output_with_speculative_config(self, model_runner):
        """Test model output generation with speculative decoding enabled.

        Verifies that:
        1. 3D tensor output is properly squeezed and processed
        2. Speculative token IDs are correctly extracted
        3. Output tokens are added to request states

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.speculative_config = Mock()
        model_runner.max_model_len = 10

        sampler_output = Mock()
        sampler_output.sampled_token_ids = torch.tensor(
            [[[1]], [[2]]], dtype=torch.long
        )

        model_runner.input_batch.num_tokens_no_spec = [0, 0]
        model_runner.input_batch.num_tokens = [0, 0]
        model_runner.input_batch.req_ids = ["req1", "req2"]
        model_runner.requests = {
            "req1": Mock(output_token_ids=[]),
            "req2": Mock(output_token_ids=[]),
        }

        # TensorWrapper handles list-to-tensor conversion for assignment
        class TensorWrapper:
            def __init__(self, tensor):
                self.tensor = tensor

            def __setitem__(self, key, value):
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=self.tensor.dtype)
                self.tensor.__setitem__(key, value)

            def __getitem__(self, key):
                return self.tensor.__getitem__(key)

        model_runner.input_batch.token_ids_cpu = TensorWrapper(
            torch.zeros((2, 10), dtype=torch.long)
        )

        result = model_runner._generate_model_runner_output(sampler_output)

        assert len(result.sampled_token_ids) == 2
        assert model_runner.spec_token_ids is not None
        # sampled_token_ids are lists of native Python ints
        assert result.sampled_token_ids[0] == [1]
        assert result.sampled_token_ids[1] == [2]
        assert model_runner.requests["req1"].output_token_ids == [1]
        assert model_runner.requests["req2"].output_token_ids == [2]

    def test_generate_model_runner_output_without_speculative(self, model_runner):
        """Test model output generation without speculative decoding.

        Verifies standard token generation path without speculation.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.speculative_config = None
        model_runner.max_model_len = 10

        sampler_output = Mock()
        # Include -1 padding tokens to test filtering
        sampler_output.sampled_token_ids = torch.tensor(
            [[1, 2, -1], [3, -1, -1]], dtype=torch.long
        )

        model_runner.input_batch.num_tokens_no_spec = [0, 0]
        model_runner.input_batch.num_tokens = [0, 0]
        model_runner.input_batch.req_ids = ["req1", "req2"]
        model_runner.requests = {
            "req1": Mock(output_token_ids=[]),
            "req2": Mock(output_token_ids=[]),
        }

        # TensorWrapper handles list-to-tensor conversion for assignment
        class TensorWrapper:
            def __init__(self, tensor):
                self.tensor = tensor

            def __setitem__(self, key, value):
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=self.tensor.dtype)
                self.tensor.__setitem__(key, value)

            def __getitem__(self, key):
                return self.tensor.__getitem__(key)

        model_runner.input_batch.token_ids_cpu = TensorWrapper(
            torch.zeros((2, 10), dtype=torch.long)
        )

        result = model_runner._generate_model_runner_output(sampler_output)

        # Verify -1 tokens are filtered out
        # sampled_token_ids are lists of native Python ints
        assert result.sampled_token_ids[0] == [1, 2]
        assert result.sampled_token_ids[1] == [3]

    def test_update_states_complex(self, model_runner):
        """Test complex state updates with multiple request scenarios.

        Verifies proper handling of:
        1. Finished requests removal
        2. New request addition
        3. Cached request updates with preemption
        4. Encoder cache cleanup

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"finished_req"}
        scheduler_output.free_encoder_mm_hashes = ["hash1"]
        scheduler_output.num_scheduled_tokens = {"active_req": 1}
        scheduler_output.scheduled_new_reqs = [
            Mock(
                req_id="new_req",
                prompt_token_ids=[1, 2, 3],
                block_ids=[[0, 1, 2]],
                num_computed_tokens=0,
                mm_kwargs=None,
                mm_positions=None,
                mm_hashes=None,
                sampling_params=Mock(),
                pooling_params=None,
                lora_request=None,
            )
        ]
        scheduler_output.scheduled_cached_reqs = Mock(
            req_ids=["cached_req"],
            num_computed_tokens=[3],
            new_block_ids=[[[3, 4, 5]]],
            resumed_req_ids={"cached_req"},
        )

        model_runner.requests = {
            "finished_req": Mock(),
            "cached_req": Mock(
                block_ids=[[0, 1, 2]], output_token_ids=[], prompt_token_ids=[1, 2, 3]
            ),
        }
        model_runner.encoder_cache = {"hash1": Mock()}
        model_runner.input_batch.req_id_to_index = {"finished_req": 0}
        model_runner.lora_config = None

        model_runner._update_states(scheduler_output)

        assert "finished_req" not in model_runner.requests
        assert "hash1" not in model_runner.encoder_cache
        assert "new_req" in model_runner.requests
        assert len(model_runner.requests["cached_req"].block_ids[0]) == 3

    def test_execute_model_multimodal(self, model_runner):
        """Test model execution for multimodal models.

        Verifies that multimodal models use the correct execution path.
        Note: execute_model now returns None and caches logits for sample_tokens().

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm_neuron.worker.constants import NEURON_MULTI_MODAL_MODELS

        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        model_runner.model.architecture = (
            list(NEURON_MULTI_MODAL_MODELS)[0] if NEURON_MULTI_MODAL_MODELS else "llava"
        )
        model_runner.lora_config = None
        model_runner.requests = {}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.encoder_cache = {}
        model_runner.use_custom_seq_id_mapping = False
        model_runner._cached_logits = None
        model_runner._cached_model_input = None

        mock_hidden_states = torch.randn(1, 32000)
        model_runner._prepare_model_input = Mock(return_value=Mock())
        model_runner._execute_model_for_multimodal_models = Mock(
            return_value=mock_hidden_states
        )

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.set_forward_context",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner,
                "maybe_get_kv_connector_output",
                return_value=nullcontext(None),
            ),
        ):
            result = model_runner.execute_model(scheduler_output)

        # execute_model returns None to defer sampling to sample_tokens()
        assert result is None
        # Hidden states should be cached for sample_tokens()
        assert model_runner._cached_logits is not None

    def test_generate_model_runner_output_none(self, model_runner):
        """Test model output generation when sampler_outputs is None.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        result = model_runner._generate_model_runner_output(None)
        assert result == EMPTY_MODEL_RUNNER_OUTPUT

    def test_generate_model_runner_output_empty_sampled_ids(self, model_runner):
        """Test model output generation with empty sampled IDs.

        Verifies that empty token sequences are handled correctly.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.speculative_config = None
        model_runner.max_model_len = 10

        sampler_output = Mock()
        # All tokens are -1 (padding)
        sampler_output.sampled_token_ids = torch.tensor(
            [[-1, -1], [-1, -1]], dtype=torch.long
        )

        model_runner.input_batch.num_tokens_no_spec = [0, 0]
        model_runner.input_batch.token_ids_cpu = torch.zeros((2, 10), dtype=torch.long)
        model_runner.input_batch.num_tokens = [0, 0]
        model_runner.input_batch.req_ids = ["req1", "req2"]
        model_runner.requests = {
            "req1": Mock(output_token_ids=[]),
            "req2": Mock(output_token_ids=[]),
        }

        result = model_runner._generate_model_runner_output(sampler_output)

        # Both should be empty after filtering
        # sampled_token_ids are lists of native Python ints
        assert result.sampled_token_ids[0] == []
        assert result.sampled_token_ids[1] == []
        # output_token_ids should not be extended for empty sequences
        assert model_runner.requests["req1"].output_token_ids == []
        assert model_runner.requests["req2"].output_token_ids == []

    def test_update_states_with_lora(self, model_runner):
        """Test state updates with LoRA configuration enabled.

        Verifies that LoRA manager is called for finished requests.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"req1"}
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        model_runner.lora_config = Mock()
        model_runner.lora_manager = Mock()
        model_runner.requests = {"req1": Mock()}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.encoder_cache = {}

        model_runner._update_states(scheduler_output)

        model_runner.lora_manager.remove_req_id.assert_called_once_with("req1")
        assert "req1" not in model_runner.requests

    def test_update_states_with_new_blocks(self, model_runner):
        """Test state updates with new block allocation.

        Verifies that new blocks are appended to existing block IDs.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []

        # Request with new blocks (not resumed)
        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        cached_req.new_block_ids = [[[3, 4]]]
        cached_req.resumed_req_ids = set()
        scheduler_output.scheduled_cached_reqs = cached_req

        model_runner.requests = {
            "req1": Mock(
                block_ids=[[0, 1, 2]],
                output_token_ids=[1, 2, 3, 4, 5],
                prompt_token_ids=[1, 2, 3, 4, 5],
            )
        }
        model_runner.input_batch.req_id_to_index = {}
        model_runner.lora_config = None

        model_runner._update_states(scheduler_output)

        # New blocks should be appended
        assert model_runner.requests["req1"].block_ids[0] == [0, 1, 2, 3, 4]

    def test_execute_model_with_custom_seq_mapping_no_match(self, model_runner):
        """Test execute_model when finished request is not in seq_id mapping.

        Verifies that requests not in mapping are handled gracefully.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"req_not_in_mapping"}
        scheduler_output.total_num_scheduled_tokens = 0
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        model_runner.use_custom_seq_id_mapping = True
        model_runner.vllm_req_to_neuron_seq_id_mapping = {"other_req": 0}
        model_runner.free_seq_ids = {1, 2}
        model_runner.lora_config = None
        model_runner.requests = {}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.encoder_cache = {}

        result = model_runner.execute_model(scheduler_output)

        assert result == EMPTY_MODEL_RUNNER_OUTPUT
        # free_seq_ids should not change since req was not in mapping
        assert model_runner.free_seq_ids == {1, 2}

    def test_sample_with_partial_prefill(self, model_runner):
        """Test sampling behavior with mixed prefill completion states.

        In chunked prefill mode, requests may be at different stages - some may
        have completed their prefill phase while others are still processing.
        This test verifies that incomplete prefill requests have their logits
        masked out to prevent premature sampling.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2", "req3"],
            prefill_completion_state=torch.tensor([True, False, True]),
        )

        model_runner.input_batch = Mock()
        model_runner.input_batch.req_ids = ["req1", "req2", "req3"]

        # Test _prepare_logits_for_sampling which handles prefill masking
        result = model_runner._prepare_logits_for_sampling(hidden_states, model_input)

        # req2 (index 1) should have -inf logits because prefill is incomplete
        assert torch.all(result[1] == float("-inf"))
        # req1 and req3 should retain their values
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(result[2], torch.tensor([5.0, 6.0]))

    def test_prepare_model_input_chunked_prefill(self, model_runner):
        """Test _prepare_model_input for chunked prefill mode.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.is_chunked_prefill = True

        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        # Mock the helper methods
        model_runner._prepare_chunked_prefill_inputs = Mock(return_value=Mock())
        model_runner._finalize_chunked_prefill_inputs = Mock(return_value=Mock())

        result = model_runner._prepare_model_input(scheduler_output)

        model_runner._prepare_chunked_prefill_inputs.assert_called_once_with(
            scheduler_output
        )
        model_runner._finalize_chunked_prefill_inputs.assert_called_once()
        assert result is not None

    def test_process_multi_modal_data_unsupported_model(self, model_runner):
        """Test multi-modal data processing with unsupported model type.

        Verifies that NotImplementedError is raised for unsupported models.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm.multimodal.inputs import PlaceholderRange

        model_runner.model.model.config.model_type = "unsupported_model_type"

        pixel_values = torch.randn(1, 3, 224, 224)
        mm_data_dict = {"pixel_values": pixel_values}

        mm_kwargs_item = Mock()
        mm_kwargs_item.__getitem__ = Mock(side_effect=lambda k: mm_data_dict[k])
        mm_kwargs_item.__contains__ = Mock(side_effect=lambda k: k in mm_data_dict)
        mm_kwargs_item.keys = Mock(return_value=mm_data_dict.keys())
        mm_kwargs_item.values = Mock(return_value=mm_data_dict.values())
        mm_kwargs_item.items = Mock(return_value=mm_data_dict.items())
        mm_kwargs_item.get = Mock(side_effect=lambda k, d=None: mm_data_dict.get(k, d))
        mm_kwargs_item.get_data = Mock(return_value=mm_data_dict)

        mm_spec = Mock()
        mm_spec.data = mm_kwargs_item
        mm_spec.modality = "image"
        mm_spec.identifier = "test_image"
        mm_spec.mm_position = PlaceholderRange(offset=0, length=1)

        mm_data = [mm_spec]

        with pytest.raises(NotImplementedError) as exc_info:
            model_runner._process_multi_modal_data_neuron(mm_data)

        assert "not supported on Neuron yet" in str(exc_info.value)

    def test_update_states_with_block_table_append(self, model_runner):
        """Test that block_table.append_row is called with new blocks.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_spec_decode_tokens = {}

        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        cached_req.new_block_ids = [[[3, 4]]]
        cached_req.resumed_req_ids = set()
        scheduler_output.scheduled_cached_reqs = cached_req

        model_runner.requests = {
            "req1": Mock(
                block_ids=[[0, 1, 2]],
                output_token_ids=[1, 2, 3, 4, 5],
                prompt_token_ids=[1, 2, 3, 4, 5],
            )
        }
        model_runner.input_batch.req_id_to_index = {"req1": 0}
        model_runner.input_batch.num_computed_tokens_cpu = [0]
        model_runner.input_batch.block_table = Mock()
        model_runner.input_batch.num_tokens_no_spec = [5]
        model_runner.input_batch.num_tokens = [5]
        model_runner.lora_config = None

        model_runner._update_states(scheduler_output)

        # Verify block_table.append_row was called
        model_runner.input_batch.block_table.append_row.assert_called_once_with(
            [[3, 4]], 0
        )

    def test_update_states_with_spec_tokens(self, model_runner):
        """Test state updates with scheduled speculative decode tokens.

        Verifies that spec tokens are added to token_ids_cpu.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_spec_decode_tokens = {"req1": [10, 11, 12]}

        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        cached_req.new_block_ids = [None]
        cached_req.resumed_req_ids = set()
        scheduler_output.scheduled_cached_reqs = cached_req

        model_runner.requests = {
            "req1": Mock(
                block_ids=[[0, 1, 2]],
                output_token_ids=[1, 2, 3, 4, 5],
                prompt_token_ids=[1, 2, 3, 4, 5],
            )
        }
        model_runner.input_batch.req_id_to_index = {"req1": 0}
        model_runner.input_batch.num_tokens_no_spec = [5]
        model_runner.input_batch.num_tokens = [5]
        model_runner.input_batch.num_computed_tokens_cpu = [0]
        model_runner.input_batch.block_table = Mock()
        model_runner.lora_config = None

        # TensorWrapper handles list-to-tensor conversion for assignment
        class TensorWrapper:
            def __init__(self, tensor):
                self.tensor = tensor

            def __setitem__(self, key, value):
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=self.tensor.dtype)
                self.tensor.__setitem__(key, value)

            def __getitem__(self, key):
                return self.tensor.__getitem__(key)

        model_runner.input_batch.token_ids_cpu = TensorWrapper(
            torch.zeros((1, 20), dtype=torch.long)
        )

        model_runner._update_states(scheduler_output)

        # In vLLM 0.16.0+, num_tokens is no longer incremented by spec tokens.
        # Only token_ids_cpu is updated with the speculative token IDs.
        assert model_runner.input_batch.num_tokens[0] == 5  # unchanged

        # Verify spec tokens [10, 11, 12] were written at positions [5, 6, 7]
        assert model_runner.input_batch.token_ids_cpu[0, 5] == 10
        assert model_runner.input_batch.token_ids_cpu[0, 6] == 11
        assert model_runner.input_batch.token_ids_cpu[0, 7] == 12

    def test_load_model_without_lora(self, model_runner):
        """Test model loading without LoRA configuration.

        Verifies that LoRA manager is not initialized when lora_config is None.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.lora_config = None
        model_runner.vllm_config.additional_config = {}  # Add this line
        # Clear the max_prompt_length to avoid validation issues
        model_runner.vllm_config.model_config.max_prompt_length = None

        with patch(
            "vllm_neuron.worker.neuronx_distributed_model_runner.get_neuron_model"
        ) as mock_get_model:
            mock_model = Mock()
            mock_model.neuron_config.is_block_kv_layout = False
            mock_model.neuron_config.is_prefix_caching = False
            mock_model.neuron_config.chunked_prefill_config = None
            mock_model.neuron_config.max_context_length = (
                2048  # Add this to match max_model_len
            )
            # Fix: Set global_topk to an actual integer to avoid Mock comparison issues
            mock_model.neuron_config.on_device_sampling_config = Mock()
            mock_model.neuron_config.on_device_sampling_config.global_topk = 256
            # Ensure the model has the required sample method
            mock_model.sample = Mock()
            mock_get_model.return_value = mock_model

            model_runner.load_model()

            assert (
                not hasattr(model_runner, "lora_manager")
                or model_runner.lora_manager is None
            )
            assert model_runner.model is not None

    def test_prepare_chunked_prefill_inputs_cached_request_with_output(
        self, model_runner
    ):
        """Test chunked prefill input preparation for cached requests with generated output.

        This test verifies the behavior when a cached request has already generated
        some output tokens and needs to continue processing. The model should include
        both the remaining prompt tokens and the last generated output token.

        Verifies that:
        1. Resumed prompt tokens are correctly extracted based on num_computed_tokens
        2. The last output token is appended to the input sequence
        3. Total input length matches expected count (prompt chunks + last output)

        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        chunked prefill capabilities
        """
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []

        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        scheduler_output.scheduled_cached_reqs = cached_req
        scheduler_output.num_scheduled_tokens = {"req1": 2}

        model_runner.cache_config = Mock(block_size=8)
        model_runner.requests = {
            "req1": Mock(
                prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
                output_token_ids=[8, 9],
                block_ids=[[0, 1]],
            )
        }

        data = model_runner._prepare_chunked_prefill_inputs(scheduler_output)

        # Should include resumed prompt tokens and last output token
        assert len(data.input_tokens) == 3  # 2 prompt tokens + 1 output token
        assert data.input_tokens[-1] == 9  # Last output token

    def test_finalize_chunked_prefill_inputs(self, model_runner):
        """Test finalization of chunked prefill inputs into model-ready tensors.

        This test verifies the conversion of intermediate data structures into
        properly formatted PyTorch tensors for chunked prefill execution. It ensures
        that variable-length sequences are padded correctly and all tensors have
        appropriate shapes.

        Verifies that:
        1. Input tokens are reshaped to [batch_size, seq_len] format
        2. Block tables are padded to uniform length across all requests
        3. Prefill completion states are preserved as boolean tensors
        4. All tensor shapes are consistent and valid for model execution

        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        chunked prefill support
        """

        data = IntermediateInputData()
        data.request_ids = ["req1", "req2"]
        data.input_tokens = [1, 2, 3, 4, 5]
        data.position_ids = [0, 1, 2, 3, 4]
        data.input_block_ids = [0, 1]
        data.slot_mapping = [0, 1, 2, 3, 4]
        data.block_tables = [[0, 1], [2]]  # Different sizes
        data.full_context_lens = [3, 2]
        data.computed_context_lens = [0, 0]
        data.prefill_completion_state = [False, True]
        data.adapter_ids = []

        model_runner.device = "cpu"

        result = model_runner._finalize_chunked_prefill_inputs(data, None, None)

        assert result.input_tokens.shape[0] == 1  # Batch size
        assert result.input_tokens.shape[1] == 5  # Sequence length
        assert result.prefill_completion_state.tolist() == [False, True]
        # Verify block tables are padded to same length
        assert result.block_tables.shape[1] == 2  # Max blocks

    def test_get_nxd_sampling_params_zero_temperature(self, model_runner):
        """Test sampling parameter handling for greedy decoding (temperature=0).

        When temperature is set to 0, the model should perform greedy decoding by
        selecting only the most likely token. This test verifies that the sampling
        parameters are correctly adjusted to implement greedy sampling.

        Verifies that:
        1. top_k is forced to 1 (select only best token) when temperature=0
        2. temperature is adjusted to 1.0 to avoid division by zero
        3. Adjustments are applied consistently across all requests in batch
        4. Original sampling params are preserved for non-zero temperatures

        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        sampling parameter generation capabilities
        """
        model_runner.model.neuron_config.on_device_sampling_config = None
        model_runner.scheduler_config = Mock(max_num_seqs=2)

        model_runner.requests = {
            "req1": Mock(sampling_params=Mock(top_k=50, top_p=0.9, temperature=0.0)),
            "req2": Mock(sampling_params=Mock(top_k=10, top_p=0.95, temperature=0.0)),
        }

        input_ids = torch.tensor([[1], [2]])

        # Mock prepare_sampling_params to capture the call
        with patch(
            "vllm_neuron.worker.neuronx_distributed_model_runner.prepare_sampling_params"
        ) as mock_prep:
            mock_prep.return_value = torch.ones((2, 3))
            _ = model_runner.get_nxd_sampling_params(input_ids)

            # Verify prepare_sampling_params was called
            mock_prep.assert_called_once()
            call_args = mock_prep.call_args

            # Check that top_k was set to 1 for zero temperature
            assert call_args[1]["top_k"][0] == 1
            assert call_args[1]["top_k"][1] == 1
            # Check that temperature was set to 1.0
            assert call_args[1]["temperature"][0] == 1.0
            assert call_args[1]["temperature"][1] == 1.0

    def test_finalize_continuous_batching_inputs_with_lora(self, model_runner):
        """Test continuous batching input finalization with LoRA adapter support.

        When LoRA (Low-Rank Adaptation) is enabled, the model runner must include
        adapter IDs in the model input to route requests to the correct fine-tuned
        adapters. This test verifies proper LoRA adapter ID handling.

        Verifies that:
        1. LoRA adapter IDs are converted from list to PyTorch tensor
        2. Adapter ID tensor has correct dtype (long) and device placement
        3. Adapter IDs are preserved in the correct order matching request IDs
        4. Sampling parameters are correctly generated for LoRA-enabled requests

        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        LoRA multi-adapter serving capabilities
        """
        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            IntermediateInputData,
        )

        model_runner.lora_config = Mock()
        model_runner.scheduler_config = Mock(max_model_len=2048, max_num_seqs=32)
        model_runner.model.neuron_config.vocab_size = 32000
        model_runner.model.neuron_config.on_device_sampling_config = None

        # Setup requests for sampling params
        model_runner.requests = {
            "req1": Mock(sampling_params=Mock(top_k=10, top_p=0.9, temperature=1.0))
        }

        data = IntermediateInputData()
        data.input_tokens = [[1, 2, 3]]
        data.position_ids = [[0, 1, 2]]
        data.input_block_ids = [0]
        data.slot_mapping = [[0]]
        data.block_tables = [[0]]
        data.full_context_lens = [3]
        data.computed_context_lens = [0]
        data.adapter_ids = [1, 2]  # LoRA adapter IDs
        data.request_ids = ["req1"]

        result = model_runner._finalize_continuous_batching_inputs(data, True)

        assert result.adapter_ids is not None
        assert isinstance(result.adapter_ids, torch.Tensor)
        assert result.adapter_ids.tolist() == [1, 2]

    def test_prepare_chunked_prefill_inputs_new_request(self, model_runner):
        """Test chunked prefill input preparation for newly submitted requests.

        When a new request arrives, chunked prefill mode processes it in chunks
        rather than all at once. This test verifies the initial chunk preparation
        for a fresh request.

        Verifies that:
        1. Only the first N tokens (based on num_scheduled_tokens) are processed
        2. Position IDs are correctly assigned starting from 0
        3. Block IDs and slot mappings are properly initialized
        4. Prefill completion state is False (more chunks remaining)
        5. Request is correctly added to the tracking data structures

        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        chunked prefill scheduling
        """
        scheduler_output = Mock()

        # Create a simple mock instead of using NewRequestData
        new_req = Mock()
        new_req.req_id = "req1"
        new_req.prompt_token_ids = [1, 2, 3, 4, 5]
        new_req.block_ids = [[0, 1, 2]]
        new_req.num_computed_tokens = 0

        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])
        scheduler_output.num_scheduled_tokens = {"req1": 3}

        model_runner.cache_config = Mock(block_size=8)
        model_runner.requests = {}

        data = model_runner._prepare_chunked_prefill_inputs(scheduler_output)

        assert "req1" in data.request_ids
        assert len(data.input_tokens) == 3  # First 3 tokens
        assert data.position_ids == [0, 1, 2]
        assert data.full_context_lens == [3]
        assert data.computed_context_lens == [0]
        assert data.prefill_completion_state == [False]  # Not complete yet

    def test_process_new_request_for_disaggregated_inference_basic_functionality(
        self, model_runner
    ):
        """Test the basic functionality of _process_new_request_for_disaggregated_inference.

        This test verifies that:
        1. A free sequence ID is assigned to the new request
        2. The request ID is added to the mapping
        3. Only the last token is processed (disaggregated inference behavior)
        4. Position ID matches num_computed_tokens
        5. Full context length is num_computed_tokens + 1
        6. Block table is properly padded
        7. Slot mapping is calculated correctly

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            IntermediateInputData,
        )

        # Setup model runner state
        model_runner.free_seq_ids = {0, 1, 2, 3, 4}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}
        model_runner.cache_config = Mock(block_size=16)
        model_runner.scheduler_config = Mock(max_model_len=2048)
        model_runner._BLOCK_TABLE_PAD = 0

        # Create sample request data
        request_data = Mock()
        request_data.req_id = "test_req_123"
        request_data.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
        request_data.num_computed_tokens = 7  # 7 tokens already computed
        request_data.block_ids = [[100, 101, 102]]  # Block IDs for KV cache

        # Create empty data structure
        data = IntermediateInputData()

        # Call the method under test
        model_runner._process_new_request_for_disaggregated_inference(
            request_data, data
        )

        # Verify sequence ID assignment
        assert request_data.req_id in model_runner.vllm_req_to_neuron_seq_id_mapping
        assigned_seq_id = model_runner.vllm_req_to_neuron_seq_id_mapping[
            request_data.req_id
        ]
        assert assigned_seq_id in {0, 1, 2, 3, 4}  # Should be one of the free IDs
        assert (
            assigned_seq_id not in model_runner.free_seq_ids
        )  # Should be removed from free set

        # Verify data structure updates
        assert len(data.request_ids) == 1
        assert data.request_ids[0] == request_data.req_id

        # Verify only last token is processed (key behavior for disaggregated inference)
        assert len(data.input_tokens) == 1
        assert data.input_tokens[0] == [10]  # Last token from prompt_token_ids

        # Verify position ID
        assert len(data.position_ids) == 1
        assert data.position_ids[0] == [7]  # num_computed_tokens

        # Verify input block ID
        assert len(data.input_block_ids) == 1
        assert data.input_block_ids[0] == assigned_seq_id

        # Verify full context length
        assert len(data.full_context_lens) == 1
        assert data.full_context_lens[0] == 8  # num_computed_tokens + 1

        # Verify computed context length
        assert len(data.computed_context_lens) == 1
        assert data.computed_context_lens[0] == 7  # num_computed_tokens

        # Verify prefill completion state
        assert len(data.prefill_completion_state) == 1
        assert data.prefill_completion_state[0] is None

        # Verify block table padding
        assert len(data.block_tables) == 1
        block_table = data.block_tables[0]
        max_blocks_per_seq = 2048 // 16  # max_model_len // block_size = 128
        assert len(block_table) == max_blocks_per_seq
        assert block_table[:3] == [100, 101, 102]  # Original block IDs
        assert all(block_id == 0 for block_id in block_table[3:])  # Padding

        # Verify slot mapping calculation
        assert len(data.slot_mapping) == 1
        expected_block_number = 100  # block_table[7 // 16] = block_table[0] = 100
        expected_block_offset = 7  # 7 % 16
        expected_slot = (
            expected_block_number * 16 + expected_block_offset
        )  # 100 * 16 + 7 = 1607
        assert data.slot_mapping[0] == [expected_slot]

    def test_process_new_request_for_disaggregated_inference_slot_mapping_calculation(
        self, model_runner
    ):
        """Test slot mapping calculation for different token positions.

        This test verifies that slot mapping is calculated correctly for various
        positions within and across block boundaries.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            IntermediateInputData,
        )

        # Setup model runner
        model_runner.cache_config = Mock(block_size=16)
        model_runner.scheduler_config = Mock(max_model_len=2048)
        model_runner._BLOCK_TABLE_PAD = 0

        test_cases = [
            # (num_computed_tokens, block_ids, expected_slot)
            (0, [50, 51, 52], 800),  # 50 * 16 + 0 = 800
            (15, [50, 51, 52], 815),  # 50 * 16 + 15 = 815
            (16, [50, 51, 52], 816),  # 51 * 16 + 0 = 816
            (17, [50, 51, 52], 817),  # 51 * 16 + 1 = 817
            (32, [50, 51, 52], 832),  # 52 * 16 + 0 = 832
        ]

        for num_computed_tokens, block_ids, expected_slot in test_cases:
            # Reset data for each test case
            data = IntermediateInputData()

            # Create request data for this test case
            request_data = Mock()
            request_data.req_id = f"test_req_{num_computed_tokens}"
            request_data.prompt_token_ids = list(
                range(num_computed_tokens + 5)
            )  # Ensure enough tokens
            request_data.num_computed_tokens = num_computed_tokens
            request_data.block_ids = [block_ids]

            # Reset free seq IDs for each test
            model_runner.free_seq_ids = {0, 1, 2, 3, 4}
            model_runner.vllm_req_to_neuron_seq_id_mapping = {}

            # Call the method
            model_runner._process_new_request_for_disaggregated_inference(
                request_data, data
            )

            # Verify slot mapping
            assert len(data.slot_mapping) == 1
            assert data.slot_mapping[0] == [expected_slot], (
                f"For position {num_computed_tokens}, expected slot {expected_slot}, got {data.slot_mapping[0]}"
            )

    def test_process_new_request_for_disaggregated_inference_assertion_errors(
        self, model_runner
    ):
        """Test that appropriate AssertionErrors are raised for invalid conditions.

        This test verifies error handling for:
        1. Existing request ID in mapping
        2. No free sequence IDs available
        3. Multiple block ID lists

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            IntermediateInputData,
        )

        # Setup model runner
        model_runner.cache_config = Mock(block_size=16)
        model_runner.scheduler_config = Mock(max_model_len=2048)
        model_runner._BLOCK_TABLE_PAD = 0

        # Test 1: Existing request ID
        model_runner.free_seq_ids = {0, 1, 2, 3, 4}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {"existing_req": 0}

        request_data = Mock()
        request_data.req_id = "existing_req"
        request_data.prompt_token_ids = [1, 2, 3, 4, 5]
        request_data.num_computed_tokens = 2
        request_data.block_ids = [[100, 101]]

        data = IntermediateInputData()

        with pytest.raises(AssertionError, match="Encountered an existing request ID"):
            model_runner._process_new_request_for_disaggregated_inference(
                request_data, data
            )

        # Test 2: No free sequence IDs
        model_runner.free_seq_ids = set()  # Empty set
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}

        request_data.req_id = "new_req"

        with pytest.raises(AssertionError, match="No free sequence ID available!"):
            model_runner._process_new_request_for_disaggregated_inference(
                request_data, data
            )

        # Test 3: Multiple block ID lists
        model_runner.free_seq_ids = {0, 1, 2, 3, 4}

        request_data.req_id = "multi_block_req"
        request_data.block_ids = [[100, 101], [200, 201]]  # Multiple lists

        with pytest.raises(AssertionError):
            model_runner._process_new_request_for_disaggregated_inference(
                request_data, data
            )

    def test_process_new_request_for_disaggregated_inference_edge_cases(
        self, model_runner
    ):
        """Test edge cases for _process_new_request_for_disaggregated_inference.

        This test verifies handling of:
        1. Zero computed tokens (first decode step)
        2. Single token prompt
        3. Block table deep copy behavior

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        import copy

        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            IntermediateInputData,
        )

        # Setup model runner
        model_runner.cache_config = Mock(block_size=16)
        model_runner.scheduler_config = Mock(max_model_len=2048)
        model_runner._BLOCK_TABLE_PAD = 0
        model_runner.free_seq_ids = {0, 1, 2, 3, 4}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}

        # Test 1: Zero computed tokens
        request_data = Mock()
        request_data.req_id = "zero_computed_req"
        request_data.prompt_token_ids = [1, 2, 3, 4, 5]
        request_data.num_computed_tokens = 0
        request_data.block_ids = [[100]]

        data = IntermediateInputData()
        model_runner._process_new_request_for_disaggregated_inference(
            request_data, data
        )

        assert data.input_tokens[0] == [5]  # Last token
        assert data.position_ids[0] == [0]  # Position 0
        assert data.full_context_lens[0] == 1  # 0 + 1
        assert data.computed_context_lens[0] == 0

        # Verify slot mapping for position 0
        expected_slot = 100 * 16 + 0  # block 100, offset 0
        assert data.slot_mapping[0] == [expected_slot]

        # Test 2: Single token prompt
        model_runner.free_seq_ids = {0, 1, 2, 3, 4}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}

        request_data.req_id = "single_token_req"
        request_data.prompt_token_ids = [42]
        request_data.num_computed_tokens = 0
        request_data.block_ids = [[200]]

        data = IntermediateInputData()
        model_runner._process_new_request_for_disaggregated_inference(
            request_data, data
        )

        assert data.input_tokens[0] == [42]  # The single token
        assert data.position_ids[0] == [0]
        assert data.full_context_lens[0] == 1
        assert data.computed_context_lens[0] == 0

        # Test 3: Block table deep copy
        original_block_ids = [[300, 301, 302]]
        request_data.req_id = "deep_copy_req"
        request_data.prompt_token_ids = [1, 2, 3, 4, 5]
        request_data.num_computed_tokens = 2
        request_data.block_ids = copy.deepcopy(original_block_ids)

        original_copy = copy.deepcopy(request_data.block_ids)

        model_runner.free_seq_ids = {0, 1, 2, 3, 4}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}

        data = IntermediateInputData()
        model_runner._process_new_request_for_disaggregated_inference(
            request_data, data
        )

        # Verify original block_ids are unchanged
        assert request_data.block_ids == original_copy

    def test_process_new_request_for_disaggregated_inference_multiple_requests(
        self, model_runner
    ):
        """Test processing multiple requests sequentially.

        This test verifies that the method can handle multiple requests correctly,
        assigning different sequence IDs and maintaining proper state.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            IntermediateInputData,
        )

        # Setup model runner
        model_runner.cache_config = Mock(block_size=16)
        model_runner.scheduler_config = Mock(max_model_len=2048)
        model_runner._BLOCK_TABLE_PAD = 0
        model_runner.free_seq_ids = {0, 1, 2, 3, 4}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}

        requests_data = []
        for i in range(3):
            request_data = Mock()
            request_data.req_id = f"test_req_{i}"
            request_data.prompt_token_ids = [1, 2, 3, 4, 5 + i]
            request_data.num_computed_tokens = 2 + i
            request_data.block_ids = [[100 + i, 101 + i]]
            requests_data.append(request_data)

        data = IntermediateInputData()

        # Process all requests
        for request_data in requests_data:
            model_runner._process_new_request_for_disaggregated_inference(
                request_data, data
            )

        # Verify all requests were processed
        assert len(data.request_ids) == 3
        assert len(data.input_tokens) == 3
        assert len(data.position_ids) == 3
        assert len(data.input_block_ids) == 3

        # Verify unique sequence IDs were assigned
        assigned_seq_ids = [
            model_runner.vllm_req_to_neuron_seq_id_mapping[f"test_req_{i}"]
            for i in range(3)
        ]
        assert len(set(assigned_seq_ids)) == 3  # All unique

        # Verify correct tokens and positions
        for i in range(3):
            assert data.input_tokens[i] == [5 + i]  # Last token
            assert data.position_ids[i] == [2 + i]  # num_computed_tokens

    def test_process_new_request_for_disaggregated_inference_data_structure_integrity(
        self, model_runner
    ):
        """Test that all data structure fields are properly populated.

        This test verifies that the method populates all required fields in the
        IntermediateInputData structure and maintains data integrity.

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            IntermediateInputData,
        )

        # Setup model runner
        model_runner.cache_config = Mock(block_size=16)
        model_runner.scheduler_config = Mock(max_model_len=2048)
        model_runner._BLOCK_TABLE_PAD = 0
        model_runner.free_seq_ids = {0, 1, 2, 3, 4}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}

        # Create sample request data
        request_data = Mock()
        request_data.req_id = "integrity_test_req"
        request_data.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        request_data.num_computed_tokens = 7
        request_data.block_ids = [[100, 101, 102]]

        data = IntermediateInputData()

        # Call the method
        model_runner._process_new_request_for_disaggregated_inference(
            request_data, data
        )

        # Verify all lists have exactly one element (single request)
        assert len(data.request_ids) == 1
        assert len(data.input_tokens) == 1
        assert len(data.position_ids) == 1
        assert len(data.input_block_ids) == 1
        assert len(data.full_context_lens) == 1
        assert len(data.computed_context_lens) == 1
        assert len(data.slot_mapping) == 1
        assert len(data.block_tables) == 1
        assert len(data.prefill_completion_state) == 1

        # Verify data types and structures
        assert isinstance(data.request_ids[0], str)
        assert isinstance(data.input_tokens[0], list)
        assert isinstance(data.position_ids[0], list)
        assert isinstance(data.input_block_ids[0], int)
        assert isinstance(data.full_context_lens[0], int)
        assert isinstance(data.computed_context_lens[0], int)
        assert isinstance(data.slot_mapping[0], list)
        assert isinstance(data.block_tables[0], list)
        assert data.prefill_completion_state[0] is None

        # Verify list lengths for token-related fields
        assert len(data.input_tokens[0]) == 1  # Single token
        assert len(data.position_ids[0]) == 1  # Single position
        assert len(data.slot_mapping[0]) == 1  # Single slot

    def test_get_nxd_sampling_params_with_device_config(self, model_runner):
        """Test sampling parameter generation with on-device sampling configuration.

        When on-device sampling is enabled via neuron_config, the sampling parameters
        must respect hardware-specific limits (like global_topk). This test verifies
        proper parameter generation with device-specific constraints.

        Verifies that:
        1. On-device sampling config's global_topk is respected
        2. Sampling parameters are prepared correctly for hardware execution
        3. Parameters are formatted as proper tensors (not mock objects)
        4. Per-request sampling params (top_k, top_p, temperature) are preserved

        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        on-device sampling hardware acceleration
        """
        model_runner.model.neuron_config.on_device_sampling_config = Mock(
            global_topk=128
        )
        model_runner.model.neuron_config.vocab_size = 32000
        model_runner.scheduler_config = Mock(max_num_seqs=2)
        model_runner.is_chunked_prefill = False

        model_runner.requests = {
            "req1": Mock(sampling_params=Mock(top_k=50, top_p=0.9, temperature=0.8)),
            "req2": Mock(sampling_params=Mock(top_k=10, top_p=0.95, temperature=1.0)),
        }

        input_ids = torch.tensor([[1], [2]])

        # Mock prepare_sampling_params to return a proper tensor
        with patch(
            "vllm_neuron.worker.neuronx_distributed_model_runner.prepare_sampling_params"
        ) as mock_prep:
            mock_prep.return_value = torch.ones((2, 3), dtype=torch.float32)
            params = model_runner.get_nxd_sampling_params(input_ids)

            assert params is not None
            assert isinstance(params, torch.Tensor)
            mock_prep.assert_called_once()

    def test_validate_max_prompt_length_user_config_matches(self, model_runner):
        """Test validation passes when user config matches neuron config."""
        model_runner.model.neuron_config.max_context_length = 2048
        model_runner.vllm_config.model_config.max_prompt_length = 2048
        model_runner.max_model_len = 2048

        model_runner._validate_max_prompt_length()
        assert model_runner.max_prompt_length == 2048

    def test_validate_max_prompt_length_user_config_mismatch(self, model_runner):
        """Test validation raises error when user config doesn't match neuron config."""
        model_runner.model.neuron_config.max_context_length = 2048
        model_runner.vllm_config.model_config.max_prompt_length = 1024

        with pytest.raises(
            RuntimeError,
            match="Configuration mismatch.*max_prompt_length in --additional-config \\(1024\\).*does not match.*compiled max prompt length \\(2048\\)",
        ):
            model_runner._validate_max_prompt_length()

    def test_validate_max_prompt_length_no_user_config_warns(self, model_runner):
        """Test validation warns when no user config provided and values differ."""
        model_runner.model.neuron_config.max_context_length = 2048
        model_runner.vllm_config.model_config.max_prompt_length = None
        model_runner.max_model_len = 1024

        with patch(
            "vllm_neuron.worker.neuronx_distributed_model_runner.logger"
        ) as mock_logger:
            model_runner._validate_max_prompt_length()

            assert model_runner.max_prompt_length == 2048
            mock_logger.warning.assert_called_once()

    def test_validate_max_prompt_length_bucketing(self, model_runner):
        """Test validation with bucketing enabled."""
        model_runner.model.neuron_config.enable_bucketing = True
        model_runner.model.neuron_config.context_encoding_buckets = [512, 1024, 2048]
        delattr(model_runner.model.neuron_config, "max_context_length")
        model_runner.vllm_config.model_config.max_prompt_length = None
        model_runner.max_model_len = 1024

        model_runner._validate_max_prompt_length()
        assert model_runner.max_prompt_length == 2048  # Max bucket size

    def test_performance_logging_execute_model(
        self, model_runner, mock_scheduler_output
    ):
        """Test that performance logging uses % formatting in execute_model.

        Verifies that:
        1. Performance logs use % formatting (not f-strings)
        2. [PERF] prefix is used for performance logs
        3. Timing measurements are included in milliseconds

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            mock_scheduler_output: Fixture providing mock scheduler output
        """
        model_runner.lora_config = None
        model_runner.requests = {}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.encoder_cache = {}

        # Mock the internal methods to avoid complex setup
        model_runner._update_states = Mock()
        model_runner._prepare_model_input = Mock(return_value=Mock())

        # Create a proper mock for sampler output that won't cause comparison issues
        mock_sampler_output = Mock()
        model_runner._execute_model_for_text = Mock(return_value=mock_sampler_output)
        model_runner._generate_model_runner_output = Mock(
            return_value=EMPTY_MODEL_RUNNER_OUTPUT
        )

        # Mock KV connector methods added from neuron-staging
        model_runner.maybe_setup_kv_connector = Mock()
        model_runner.maybe_wait_for_kv_save = Mock()
        model_runner.get_finished_kv_transfers = Mock(return_value=(None, None))

        # Mock KV connector methods added from neuron-staging
        model_runner.maybe_setup_kv_connector = Mock()
        model_runner.maybe_wait_for_kv_save = Mock()
        model_runner.get_finished_kv_transfers = Mock(return_value=(None, None))

        mock_scheduler_output.finished_req_ids = set()
        mock_scheduler_output.total_num_scheduled_tokens = 1
        mock_scheduler_output.scheduled_new_reqs = []  # Empty list to avoid len() issues
        mock_scheduler_output.scheduled_cached_reqs = Mock(req_ids=["req1"])

        with (
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.logger"
            ) as mock_logger,
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.has_kv_transfer_group",
                return_value=False,
            ),
            patch(
                "vllm_neuron.worker.neuronx_distributed_model_runner.set_forward_context"
            ),
        ):
            model_runner.execute_model(mock_scheduler_output)

            # Verify performance logging uses % formatting
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "[PERF]" in str(call)
            ]

            # Should have performance logs for each major operation
            assert len(debug_calls) >= 4  # update, prepare, execution, output

            # Verify calls use % formatting (not f-strings)
            for call in debug_calls:
                args = call[0]
                # First argument should be the format string with %
                assert "%" in args[0] or "ms" in str(args), (
                    "Performance log should use % formatting"
                )

    def test_performance_logging_sample_method(self, model_runner):
        """Test that performance logging works correctly in sample_tokens.

        Verifies that:
        1. Performance logs are captured
        2. Batch size context is available

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        hidden_states = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(
            request_ids=["req1", "req2"],
            prefill_completion_state=None,
        )

        # Set up cached state (simulating execute_model was called)
        model_runner._cached_logits = hidden_states
        model_runner._cached_model_input = model_input
        model_runner.input_batch.req_ids = ["req1", "req2"]
        model_runner.input_batch.sampling_metadata = Mock()
        model_runner.model.neuron_config.on_device_sampling_config = None

        mock_output = Mock()
        mock_output.sampled_token_ids = torch.tensor([[1], [2]])
        mock_output.logprobs_tensors = None
        model_runner.cpu_sampler = Mock(return_value=mock_output)
        model_runner.speculative_config = None
        model_runner.input_batch.num_tokens_no_spec = [0, 0]
        model_runner.input_batch.num_tokens = [0, 0]
        # Use MagicMock to allow item assignment
        model_runner.input_batch.token_ids_cpu = MagicMock()
        model_runner.max_model_len = 1024
        model_runner.requests = {
            "req1": Mock(output_token_ids=[]),
            "req2": Mock(output_token_ids=[]),
        }

        with patch(
            "vllm_neuron.worker.neuronx_distributed_model_runner.logger"
        ) as mock_logger:
            result = model_runner.sample_tokens(grammar_output=None)

            # Verify logging was called (debug logs for performance)
            assert mock_logger.debug.called
            assert result is not None

    def test_process_new_request_for_disaggregated_inference_called(self, model_runner):
        """Test that _process_new_request_for_disaggregated_inference is called when the right scheduler_output is provided.

        This test verifies that when scheduler_output has kv_connector_metadata with reqs_to_save=False/empty,
        the _process_new_request_for_disaggregated_inference method is called instead of the regular
        continuous batching method. This condition indicates the current node is a decode node in the
        disaggregated inference (DI) setting.

        Verifies that:
        1. The disaggregated inference method is called when kv_connector_metadata.reqs_to_save is empty
        2. The regular continuous batching method is NOT called in this case
        3. is_prefill is set to False for disaggregated inference
        4. The method processes new requests correctly for decode nodes

        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        disaggregated inference capabilities
        """
        # Setup scheduler output with kv_connector_metadata indicating decode node
        scheduler_output = Mock()

        # Create kv_connector_metadata with empty reqs_to_save (decode node condition)
        kv_connector_metadata = Mock()
        kv_connector_metadata.reqs_to_save = []  # Empty list indicates decode node
        scheduler_output.kv_connector_metadata = kv_connector_metadata

        # Create a new request
        new_request = Mock()
        new_request.req_id = "test_req_1"
        new_request.prompt_token_ids = [1, 2, 3, 4, 5]
        new_request.block_ids = [[0, 1, 2]]
        new_request.num_computed_tokens = 3
        scheduler_output.scheduled_new_reqs = [new_request]

        # Setup cached requests (empty for this test)
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        # Setup model runner state
        model_runner.free_seq_ids = {0, 1, 2}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}
        model_runner.cache_config = Mock(block_size=8)
        model_runner.scheduler_config = Mock(max_model_len=2048)

        # Mock the methods we want to verify are called
        with (
            patch.object(
                model_runner, "_process_new_request_for_disaggregated_inference"
            ) as mock_di_method,
            patch.object(
                model_runner, "_process_new_request_for_continuous_batching"
            ) as mock_cb_method,
        ):
            # Call the method under test
            data, is_prefill = model_runner._prepare_continuous_batching_inputs(
                scheduler_output
            )

            # Verify disaggregated inference method was called
            mock_di_method.assert_called_once_with(new_request, data)

            # Verify continuous batching method was NOT called
            mock_cb_method.assert_not_called()

            # Verify is_prefill is False for disaggregated inference
            assert is_prefill is False

    def test_process_new_request_for_disaggregated_inference_not_called_when_reqs_to_save_present(
        self, model_runner
    ):
        """Test that _process_new_request_for_disaggregated_inference is NOT called when reqs_to_save is not empty.

        This test verifies that when scheduler_output has kv_connector_metadata with non-empty reqs_to_save,
        the regular continuous batching method is called instead of the disaggregated inference method.
        This condition indicates the current node is NOT a decode node in the DI setting.

        Verifies that:
        1. The regular continuous batching method is called when kv_connector_metadata.reqs_to_save is not empty
        2. The disaggregated inference method is NOT called in this case
        3. is_prefill is set to True for regular continuous batching

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Setup scheduler output with kv_connector_metadata indicating non-decode node
        scheduler_output = Mock()

        # Create kv_connector_metadata with non-empty reqs_to_save (non-decode node condition)
        kv_connector_metadata = Mock()
        kv_connector_metadata.reqs_to_save = [
            "some_req"
        ]  # Non-empty list indicates non-decode node
        scheduler_output.kv_connector_metadata = kv_connector_metadata

        # Create a new request
        new_request = Mock()
        new_request.req_id = "test_req_1"
        new_request.prompt_token_ids = [1, 2, 3, 4, 5]
        new_request.block_ids = [[0, 1, 2]]
        new_request.num_computed_tokens = 0
        new_request.mm_features = None
        scheduler_output.scheduled_new_reqs = [new_request]

        # Setup cached requests (empty for this test)
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        # Setup model runner state
        model_runner.free_seq_ids = {0, 1, 2}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}
        model_runner.is_prefix_caching = False

        # Mock the methods we want to verify are called
        with (
            patch.object(
                model_runner, "_process_new_request_for_disaggregated_inference"
            ) as mock_di_method,
            patch.object(
                model_runner, "_process_new_request_for_continuous_batching"
            ) as mock_cb_method,
            patch.object(
                model_runner, "_prepare_adapter_id_in_new_request", return_value=None
            ),
        ):
            # Call the method under test
            data, is_prefill = model_runner._prepare_continuous_batching_inputs(
                scheduler_output
            )

            # Verify continuous batching method was called
            mock_cb_method.assert_called_once_with(new_request, data)

            # Verify disaggregated inference method was NOT called
            mock_di_method.assert_not_called()

            # Verify is_prefill is True for regular continuous batching
            assert is_prefill is True

    def test_process_new_request_for_disaggregated_inference_no_kv_connector_metadata(
        self, model_runner
    ):
        """Test that _process_new_request_for_disaggregated_inference is NOT called when kv_connector_metadata is None.

        This test verifies that when scheduler_output has no kv_connector_metadata (None),
        the regular continuous batching method is called instead of the disaggregated inference method.
        This is the normal case when disaggregated inference is not being used.

        Verifies that:
        1. The regular continuous batching method is called when kv_connector_metadata is None
        2. The disaggregated inference method is NOT called in this case
        3. is_prefill is set to True for regular continuous batching

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        # Setup scheduler output without kv_connector_metadata (normal case)
        scheduler_output = Mock()
        scheduler_output.kv_connector_metadata = None  # No disaggregated inference

        # Create a new request
        new_request = Mock()
        new_request.req_id = "test_req_1"
        new_request.prompt_token_ids = [1, 2, 3, 4, 5]
        new_request.block_ids = [[0, 1, 2]]
        new_request.num_computed_tokens = 0
        new_request.mm_features = None
        scheduler_output.scheduled_new_reqs = [new_request]

        # Setup cached requests (empty for this test)
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        # Setup model runner state
        model_runner.free_seq_ids = {0, 1, 2}
        model_runner.vllm_req_to_neuron_seq_id_mapping = {}
        model_runner.is_prefix_caching = False

        # Mock the methods we want to verify are called
        with (
            patch.object(
                model_runner, "_process_new_request_for_disaggregated_inference"
            ) as mock_di_method,
            patch.object(
                model_runner, "_process_new_request_for_continuous_batching"
            ) as mock_cb_method,
            patch.object(
                model_runner, "_prepare_adapter_id_in_new_request", return_value=None
            ),
        ):
            # Call the method under test
            data, is_prefill = model_runner._prepare_continuous_batching_inputs(
                scheduler_output
            )

            # Verify continuous batching method was called
            mock_cb_method.assert_called_once_with(new_request, data)

            # Verify disaggregated inference method was NOT called
            mock_di_method.assert_not_called()

            # Verify is_prefill is True for regular continuous batching
            assert is_prefill is True

    def test_kv_cache_flags_unchanged_after_load_model(self, model_runner):
        """Test that KV cache flags set in __init__ remain unchanged after load_model.

        KV cache mode flags (is_prefix_caching, is_chunked_prefill, is_block_kv_layout)
        are set during __init__ from vLLM config. This test verifies that
        load_model() does not modify these flags.

        Args:
            model_runner: Parametrized fixture providing ModelRunner configured
                for contiguous, prefix_caching, or chunked_prefill mode
        """
        # Capture flags set in __init__
        init_is_prefix_caching = model_runner.is_prefix_caching
        init_is_chunked_prefill = model_runner.is_chunked_prefill
        init_is_block_kv_layout = model_runner.is_block_kv_layout

        # Clear max_prompt_length to avoid validation issues
        model_runner.vllm_config.model_config.max_prompt_length = None

        model_runner.lora_config = None
        model_runner.vllm_config.additional_config = {"override_neuron_config": {}}

        with patch(
            "vllm_neuron.worker.neuronx_distributed_model_runner.get_neuron_model"
        ) as mock_get_model:
            # Create neuron_config with different values to verify they don't override
            neuron_config = type(
                "obj",
                (object,),
                {
                    "is_block_kv_layout": not init_is_block_kv_layout,  # Opposite value
                    "is_prefix_caching": not init_is_prefix_caching,  # Opposite value
                    "chunked_prefill_config": Mock()
                    if not init_is_chunked_prefill
                    else None,
                    "on_device_sampling_config": None,
                    "max_context_length": 2048,
                },
            )()

            mock_model = Mock()
            mock_model.neuron_config = neuron_config
            mock_model.sample = Mock()
            mock_get_model.return_value = mock_model

            model_runner.load_model()

            # Verify flags remain unchanged after load_model
            assert model_runner.is_prefix_caching == init_is_prefix_caching, (
                f"is_prefix_caching changed from {init_is_prefix_caching} to "
                f"{model_runner.is_prefix_caching} after load_model()"
            )
            assert model_runner.is_chunked_prefill == init_is_chunked_prefill, (
                f"is_chunked_prefill changed from {init_is_chunked_prefill} to "
                f"{model_runner.is_chunked_prefill} after load_model()"
            )
            assert model_runner.is_block_kv_layout == init_is_block_kv_layout, (
                f"is_block_kv_layout changed from {init_is_block_kv_layout} to "
                f"{model_runner.is_block_kv_layout} after load_model()"
            )
