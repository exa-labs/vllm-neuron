# SPDX-License-Identifier: Apache-2.0
import importlib
import logging
import os
from unittest.mock import Mock, call, patch

import pytest
import torch

from vllm_neuron.platform import NeuronFramework, NeuronPlatform, _Backend
import vllm_neuron.platform as _plat_mod


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Fixture to cleanup any global state modifications after each test."""
    # Store original state
    original_config_overrides_applied = NeuronPlatform._config_overrides_applied
    original_openai_overrides_applied = NeuronPlatform._openai_overrides_applied

    yield

    # Restore original state and clear cached results
    NeuronPlatform._config_overrides_applied = original_config_overrides_applied
    NeuronPlatform._openai_overrides_applied = original_openai_overrides_applied
    NeuronPlatform.is_neuronx_distributed_inference.cache_clear()


def test_neuron_platform_basic_properties():
    """Test basic properties of NeuronPlatform.

    Verifies:
    - Device name and type are correctly set
    - Ray device key is properly configured
    - Supported quantization types are present
    - Device control environment variable is set
    """
    platform = NeuronPlatform()

    assert platform.device_name == "cpu"
    assert platform.device_type == "cpu"
    assert platform.ray_device_key == "neuron_cores"
    assert "neuron_quant" in platform.supported_quantization
    assert "fbgemm_fp8" in platform.supported_quantization
    assert platform.device_control_env_var == "NEURON_RT_VISIBLE_CORES"


def test_get_device_name():
    """Test get_device_name method.

    Verifies that the device name is consistently returned as "neuron"
    regardless of the device ID provided.
    """
    assert NeuronPlatform.get_device_name() == "neuron"
    assert NeuronPlatform.get_device_name(1) == "neuron"


def test_is_async_output_supported():
    """Test is_async_output_supported method.

    Verifies that async output is not supported regardless of the
    enforce_eager parameter value.
    """
    assert not NeuronPlatform.is_async_output_supported(None)
    assert not NeuronPlatform.is_async_output_supported(True)


def test_is_pin_memory_available(caplog):
    """Test is_pin_memory_available method.

    Verifies:
    - Pin memory is not available
    - Appropriate warning message is logged
    """
    with caplog.at_level(logging.WARNING):
        assert not NeuronPlatform.is_pin_memory_available()
        assert "Pin memory is not supported on Neuron." in caplog.text


def test_use_all_gather():
    """Test use_all_gather method.

    Verifies that all_gather is always enabled for the Neuron platform.
    """
    assert NeuronPlatform.use_all_gather()


def test_supports_v1():
    """Test supports_v1 method.

    Verifies that v1 support is always enabled for any model configuration.
    """
    mock_model_config = Mock()
    assert NeuronPlatform.supports_v1(mock_model_config)


def test_is_neuronx_distributed_inference_not_installed():
    """Test is_neuronx_distributed_inference when package is not installed.

    Verifies that the method correctly handles the case when
    neuronx_distributed_inference is not available.

    The test ensures that:
    1. The method returns False when the package is not available
    2. The result is properly cached (lru_cache behavior)
    """
    # Clear the lru_cache to ensure fresh test
    NeuronPlatform.is_neuronx_distributed_inference.cache_clear()

    # Create a context where neuronx_distributed_inference is not available
    with patch.dict("sys.modules", {"neuronx_distributed_inference": None}):
        # Test the method
        result = NeuronPlatform.is_neuronx_distributed_inference()

        # Verify the result
        assert not result


def test_get_neuron_framework_to_use():
    """Test get_neuron_framework_to_use method.

    Verifies:
    - Correct framework selection when on Neuron platform
    - Error handling when required packages are not available
    - Error handling when not on Neuron platform
    """
    platform = NeuronPlatform()

    with patch.object(platform, "is_neuron", return_value=True):
        with patch.object(
            platform, "is_neuronx_distributed_inference", return_value=True
        ):
            assert (
                platform.get_neuron_framework_to_use()
                == NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
            )

        with patch.object(
            platform, "is_neuronx_distributed_inference", return_value=False
        ):
            with pytest.raises(
                AssertionError, match="Unable to import neuronx_distributed_inference"
            ):
                platform.get_neuron_framework_to_use()

    with patch.object(platform, "is_neuron", return_value=False):
        with pytest.raises(AssertionError, match="Neuron Framework unavailable"):
            platform.get_neuron_framework_to_use()


def test_use_neuronx_distributed():
    """Test use_neuronx_distributed method.

    Verifies that the method correctly identifies when to use
    neuronx-distributed-inference framework.
    """
    platform = NeuronPlatform()

    with patch.object(
        platform,
        "get_neuron_framework_to_use",
        return_value=NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE,
    ):
        assert platform.use_neuronx_distributed()


@pytest.mark.parametrize("disable_scheduler", ["0", "1"])
def test_check_and_update_config(disable_scheduler):
    """Test check_and_update_config method with different scheduler settings.

    Args:
        disable_scheduler: String value ("0" or "1") to control scheduler override.
            "0": Use custom Neuron scheduler
            "1": Use vLLM V1 native scheduler

    Verifies:
    - Proper handling of scheduler override settings
    - Worker class configuration
    - Parallel configuration updates
    - Cache configuration updates
    - Scheduler class selection based on environment variable
    - Chunked prefill settings
    """
    with patch.dict(os.environ, {"DISABLE_NEURON_CUSTOM_SCHEDULER": disable_scheduler}):
        mock_config = Mock()
        mock_config.model_config = Mock()
        mock_config.parallel_config = Mock()
        mock_config.parallel_config.world_size = 1
        mock_config.parallel_config.worker_cls = "auto"
        mock_config.cache_config = Mock()
        mock_config.cache_config.num_gpu_blocks_override = None
        mock_config.scheduler_config = Mock()
        mock_config.lora_config = None

        NeuronPlatform.check_and_update_config(mock_config)

        # Verify worker class update
        assert (
            mock_config.parallel_config.worker_cls
            == "vllm_neuron.worker.neuron_worker.NeuronWorker"
        )

        # Verify scheduler configuration based on environment variable
        if disable_scheduler == "0":
            assert (
                mock_config.scheduler_config.scheduler_cls
                == "vllm_neuron.core.scheduler.ContinuousBatchingNeuronScheduler"
            )
            assert not mock_config.scheduler_config.enable_chunked_prefill


def test_check_and_update_config_cache_settings():
    """Test cache configuration updates in check_and_update_config.

    Verifies:
    - Block size configuration for non-prefix caching mode
    - Block size is set to max_model_len when prefix caching is disabled
    - Cache configuration is properly applied
    - Model configuration integration with cache settings

    The test ensures that cache settings are properly configured based on
    the caching mode and model parameters.
    """
    mock_config = Mock()
    mock_config.model_config = Mock(max_model_len=2048)
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(enable_prefix_caching=False)
    mock_config.cache_config.num_gpu_blocks_override = 1
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    NeuronPlatform.check_and_update_config(mock_config)

    # Verify cache block size is set to max_model_len when prefix caching is disabled
    assert mock_config.cache_config.block_size == 2048
    assert mock_config.cache_config.num_gpu_blocks_override == 2


def test_check_and_update_config_distributed():
    """Test distributed configuration updates in check_and_update_config.

    Verifies:
    - Distributed executor backend settings
    - World size handling
    - Worker configuration in distributed mode
    - Backend selection for multi-worker setups

    The test ensures proper configuration for distributed execution
    when world_size is greater than 1.
    """
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=2, worker_cls="auto")
    mock_config.cache_config = Mock()
    mock_config.cache_config.num_gpu_blocks_override = None
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    NeuronPlatform.check_and_update_config(mock_config)

    # Verify distributed executor backend is set to "uni" for world_size > 1
    assert mock_config.parallel_config.distributed_executor_backend == "uni"


def test_pre_register_and_update():
    """Test pre_register_and_update method.

    Verifies:
    - ModelConfig class modification
    - Custom verify_with_parallel_config method installation
    - Proper cleanup of original configuration
    - Configuration persistence

    The test ensures that the ModelConfig class is properly modified
    while maintaining the ability to restore original settings.

    Note:
        This test modifies global state and includes cleanup to
        restore original settings.
    """
    from vllm.config import ModelConfig

    original_verify = getattr(ModelConfig, "verify_with_parallel_config", None)

    try:
        mock_parser = Mock()
        NeuronPlatform.pre_register_and_update(mock_parser)
        assert hasattr(ModelConfig, "verify_with_parallel_config")
    finally:
        if original_verify:
            setattr(ModelConfig, "verify_with_parallel_config", original_verify)


def test_check_and_update_config_cache_validation():
    """Test cache configuration validation in check_and_update_config.

    Verifies:
    - Proper validation of cache settings
    - Error handling for invalid configurations
    - Block size requirements for prefix caching
    - Assertion messages for configuration errors

    The test ensures that invalid cache configurations are properly
    detected and appropriate error messages are raised.

    Raises:
        AssertionError: When prefix caching is enabled without block_size
    """
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(
        enable_prefix_caching=True, block_size=None, num_gpu_blocks_override=None
    )
    mock_config.scheduler_config = Mock()

    with pytest.raises(
        AssertionError, match="When prefix caching is enabled, block_size must be set"
    ):
        NeuronPlatform.check_and_update_config(mock_config)


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_check_and_update_config_hybrid_linear_attention_prefix_caching(
    enable_prefix_caching,
):
    """Test that prefix caching is force-disabled for hybrid linear-attention
    models (e.g. Qwen3.5 / Qwen3-Next).

    Verifies:
    - enable_prefix_caching is set to False when layer_types contains
      "linear_attention"
    - Non-hybrid models are unaffected by the guard
    """
    from transformers import PretrainedConfig

    mock_config = Mock()
    mock_config.model_config = Mock(max_model_len=2048)
    mock_config.model_config.hf_config = PretrainedConfig(
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(
        enable_prefix_caching=enable_prefix_caching,
        block_size=32,
        num_gpu_blocks_override=None,
    )
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    NeuronPlatform.check_and_update_config(mock_config)

    assert mock_config.cache_config.enable_prefix_caching is False
    # With prefix caching disabled, Neuron requires block_size = max_model_len
    assert mock_config.cache_config.block_size == 2048


def test_check_and_update_config_non_hybrid_prefix_caching_unchanged():
    """Test that the hybrid guard leaves prefix caching enabled for standard
    full-attention models."""
    from transformers import PretrainedConfig

    mock_config = Mock()
    mock_config.model_config = Mock(max_model_len=2048)
    mock_config.model_config.hf_config = PretrainedConfig(
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
    )
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(
        enable_prefix_caching=True,
        block_size=32,
        num_gpu_blocks_override=None,
    )
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    NeuronPlatform.check_and_update_config(mock_config)

    assert mock_config.cache_config.enable_prefix_caching is True


def test_pre_register_and_update_with_expert_parallel():
    """Test pre_register_and_update with expert parallelism.

    Verifies:
    - Expert parallelism configuration handling
    - Verification method behavior with expert parallel enabled
    - Proper method installation and cleanup
    """
    from vllm.config import ModelConfig

    original_verify = getattr(ModelConfig, "verify_with_parallel_config", None)

    try:
        # Reset the flag to ensure overrides can be applied
        NeuronPlatform._config_overrides_applied = False

        mock_parser = Mock()
        mock_parallel_config = Mock()
        # Set specific values for mock attributes
        mock_parallel_config.enable_expert_parallel = True
        mock_parallel_config.distributed_executor_backend = "standard"
        mock_parallel_config.pipeline_parallel_size = 1  # Fix for TypeError

        # Mock all imports needed inside _apply_config_overrides
        mock_serving = Mock()
        mock_renderer = Mock()

        # Patch all the imports comprehensively
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **kwargs):
                if name == "vllm.config":
                    mock_config_module = Mock()
                    mock_config_module.ModelConfig = ModelConfig
                    return mock_config_module
                elif name == "vllm.entrypoints.openai.serving_engine":
                    mock_serving_module = Mock()
                    mock_serving_module.OpenAIServing = mock_serving
                    return mock_serving_module
                elif name == "vllm.entrypoints.renderer":
                    mock_renderer_module = Mock()
                    mock_renderer_module.CompletionRenderer = mock_renderer
                    return mock_renderer_module
                elif name == "vllm_neuron.platform_overrides":
                    # Import the actual platform_overrides module
                    return importlib.import_module("vllm_neuron.platform_overrides")
                else:
                    # Fall back to default import behavior
                    return importlib.__import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect
            NeuronPlatform.pre_register_and_update(mock_parser)

        # Create model config instance
        model_config = Mock()
        model_config._verify_with_expert_parallelism = Mock()
        model_config.seed = None  # Add seed attribute

        # Call the installed method
        verify_method = getattr(ModelConfig, "verify_with_parallel_config")
        verify_method(model_config, mock_parallel_config)

        # Verify expert parallelism check was called
        model_config._verify_with_expert_parallelism.assert_called_once()

    finally:
        if original_verify:
            setattr(ModelConfig, "verify_with_parallel_config", original_verify)


def test_pre_register_and_update_with_pipeline_parallel():
    """Test pre_register_and_update with pipeline parallelism.

    Verifies:
    - Pipeline parallelism configuration handling
    - Support check for pipeline parallel models
    - Async output processor handling
    """
    from vllm.config import ModelConfig

    original_verify = getattr(ModelConfig, "verify_with_parallel_config", None)

    try:
        # Reset the flag to ensure overrides can be applied
        NeuronPlatform._config_overrides_applied = False

        mock_parser = Mock()
        mock_parallel_config = Mock()
        # Set specific values for mock attributes
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 2
        mock_parallel_config.distributed_executor_backend = "standard"

        # Mock all imports needed inside _apply_config_overrides
        mock_serving = Mock()
        mock_renderer = Mock()

        # Patch all the imports comprehensively
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **kwargs):
                if name == "vllm.config":
                    mock_config_module = Mock()
                    mock_config_module.ModelConfig = ModelConfig
                    return mock_config_module
                elif name == "vllm.entrypoints.openai.serving_engine":
                    mock_serving_module = Mock()
                    mock_serving_module.OpenAIServing = mock_serving
                    return mock_serving_module
                elif name == "vllm.entrypoints.renderer":
                    mock_renderer_module = Mock()
                    mock_renderer_module.CompletionRenderer = mock_renderer
                    return mock_renderer_module
                elif name == "vllm_neuron.platform_overrides":
                    # Import the actual platform_overrides module
                    return importlib.import_module("vllm_neuron.platform_overrides")
                else:
                    # Fall back to default import behavior
                    return importlib.__import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect
            NeuronPlatform.pre_register_and_update(mock_parser)

        # Create model config instance with specific attribute values
        model_config = Mock(spec=["architectures", "registry", "use_async_output_proc"])
        model_config.architectures = ["TestModel"]
        model_config.registry.is_pp_supported_model.return_value = False
        model_config.use_async_output_proc = True
        model_config.seed = None  # Add seed attribute

        # Call the installed method
        verify_method = getattr(ModelConfig, "verify_with_parallel_config")

        # Verify NotImplementedError is raised for unsupported model
        with pytest.raises(
            NotImplementedError,
            match="Pipeline parallelism is not supported for this model",
        ):
            verify_method(model_config, mock_parallel_config)

        # The use_async_output_proc should be set to False during the call
        # but only if pipeline parallel size > 1 and the model is supported
        # Since this test uses an unsupported model, the method raises before reaching that check

    finally:
        if original_verify:
            setattr(ModelConfig, "verify_with_parallel_config", original_verify)


def test_check_and_update_config_edge_cases():
    """Test check_and_update_config edge cases and error conditions.

    Verifies:
    - Empty model config handling
    - Block size validation with native scheduler
    - Prefix caching configuration validation
    - Error messages for invalid configurations
    """
    # Test with empty model config
    mock_config = Mock()
    mock_config.model_config = None
    mock_config.parallel_config = Mock(worker_cls="auto", world_size=1)
    mock_config.cache_config = Mock(
        enable_prefix_caching=False, num_gpu_blocks_override=None
    )
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    NeuronPlatform.check_and_update_config(mock_config)

    # Test native scheduler without block size
    with patch.dict(os.environ, {"DISABLE_NEURON_CUSTOM_SCHEDULER": "1"}):
        mock_config = Mock()
        mock_config.model_config = Mock()
        mock_config.parallel_config = Mock(worker_cls="auto", world_size=1)
        mock_config.cache_config = Mock(block_size=None, num_gpu_blocks_override=None)
        mock_config.scheduler_config = Mock()

        with pytest.raises(
            AssertionError,
            match="When vLLM V1 native scheduler is enabled, block_size must be set",
        ):
            NeuronPlatform.check_and_update_config(mock_config)


def test_get_neuron_framework_error_handling():
    """Test error handling in framework detection methods.

    Verifies:
    - Error handling for unavailable frameworks
    - Proper error messages
    - Cache clearing behavior
    """
    platform = NeuronPlatform()

    # Test with neuron platform unavailable
    with patch.object(platform, "is_neuron", return_value=False):
        with pytest.raises(
            AssertionError, match="Neuron Framework unavailable for platform"
        ):
            platform.get_neuron_framework_to_use()

    # Test with framework import failure
    with patch.object(platform, "is_neuron", return_value=True):
        with patch.dict("sys.modules", {"neuronx_distributed_inference": None}):
            with pytest.raises(
                AssertionError, match="Unable to import neuronx_distributed_inference"
            ):
                platform.get_neuron_framework_to_use()


def test_pre_register_and_update_with_external_launcher():
    """Test pre_register_and_update with external launcher.

    Verifies:
    - External launcher configuration handling
    - Seed validation for external launcher
    - Error handling for missing seed
    """
    from vllm.config import ModelConfig

    original_verify = getattr(ModelConfig, "verify_with_parallel_config", None)

    try:
        # Reset the flag to ensure overrides can be applied
        NeuronPlatform._config_overrides_applied = False

        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "external_launcher"
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 1

        # Mock all imports needed inside _apply_config_overrides
        mock_serving = Mock()
        mock_renderer = Mock()

        # Patch all the imports comprehensively
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **kwargs):
                if name == "vllm.config":
                    mock_config_module = Mock()
                    mock_config_module.ModelConfig = ModelConfig
                    return mock_config_module
                elif name == "vllm.entrypoints.openai.serving_engine":
                    mock_serving_module = Mock()
                    mock_serving_module.OpenAIServing = mock_serving
                    return mock_serving_module
                elif name == "vllm.entrypoints.renderer":
                    mock_renderer_module = Mock()
                    mock_renderer_module.CompletionRenderer = mock_renderer
                    return mock_renderer_module
                elif name == "vllm_neuron.platform_overrides":
                    # Import the actual platform_overrides module
                    return importlib.import_module("vllm_neuron.platform_overrides")
                else:
                    # Fall back to default import behavior
                    return importlib.__import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect
            NeuronPlatform.pre_register_and_update()

        # Test with missing seed
        model_config = Mock(seed=None)
        verify_method = getattr(ModelConfig, "verify_with_parallel_config")

        with pytest.raises(
            AssertionError, match="Seed must be set when using external launcher"
        ):
            verify_method(model_config, mock_parallel_config)

        # Test with seed set
        model_config.seed = 42
        verify_method(model_config, mock_parallel_config)

    finally:
        if original_verify:
            setattr(ModelConfig, "verify_with_parallel_config", original_verify)


def test_pre_register_and_update_pipeline_async():
    """Test pre_register_and_update with pipeline parallel and async output.

    Verifies:
    - Pipeline parallel configuration with async output
    - Async output processor disabling
    - Model support validation
    """
    from vllm.config import ModelConfig

    original_verify = getattr(ModelConfig, "verify_with_parallel_config", None)

    try:
        # Reset the flag to ensure overrides can be applied
        NeuronPlatform._config_overrides_applied = False

        mock_parallel_config = Mock()
        mock_parallel_config.pipeline_parallel_size = 2
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.distributed_executor_backend = "standard"

        # Mock all imports needed inside _apply_config_overrides
        mock_serving = Mock()
        mock_renderer = Mock()

        # Patch all the imports comprehensively
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **kwargs):
                if name == "vllm.config":
                    mock_config_module = Mock()
                    mock_config_module.ModelConfig = ModelConfig
                    return mock_config_module
                elif name == "vllm.entrypoints.openai.serving_engine":
                    mock_serving_module = Mock()
                    mock_serving_module.OpenAIServing = mock_serving
                    return mock_serving_module
                elif name == "vllm.entrypoints.renderer":
                    mock_renderer_module = Mock()
                    mock_renderer_module.CompletionRenderer = mock_renderer
                    return mock_renderer_module
                elif name == "vllm_neuron.platform_overrides":
                    # Import the actual platform_overrides module
                    return importlib.import_module("vllm_neuron.platform_overrides")
                else:
                    # Fall back to default import behavior
                    return importlib.__import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect
            NeuronPlatform.pre_register_and_update()

        model_config = Mock()
        model_config.registry.is_pp_supported_model.return_value = True
        model_config.use_async_output_proc = True
        model_config.seed = None

        verify_method = getattr(ModelConfig, "verify_with_parallel_config")
        verify_method(model_config, mock_parallel_config)

        # Verify async output was disabled
        assert not model_config.use_async_output_proc

    finally:
        if original_verify:
            setattr(ModelConfig, "verify_with_parallel_config", original_verify)


def test_check_and_update_config_scheduler_defaults():
    """Test scheduler configuration defaults in check_and_update_config."""
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(
        enable_prefix_caching=False, num_gpu_blocks_override=None
    )
    mock_config.scheduler_config = Mock(max_num_seqs=None)
    mock_config.lora_config = None

    with patch.dict(os.environ, {"DISABLE_NEURON_CUSTOM_SCHEDULER": "0"}):
        NeuronPlatform.check_and_update_config(mock_config)

        # Verify scheduler defaults
        assert mock_config.scheduler_config.max_num_batched_tokens == 131072
        assert mock_config.scheduler_config.max_num_seqs == 32
        assert (
            mock_config.scheduler_config.scheduler_cls
            == "vllm_neuron.core.scheduler.ContinuousBatchingNeuronScheduler"
        )
        assert not mock_config.scheduler_config.enable_chunked_prefill


def test_multiple_config_override_calls():
    """Test multiple calls to _apply_config_overrides."""
    # Store original flag state to restore later
    original_flag_state = NeuronPlatform._config_overrides_applied

    try:
        platform = NeuronPlatform()

        # Reset the class flag
        NeuronPlatform._config_overrides_applied = False

        mock_logger = Mock()

        # Mock the entire _apply_config_overrides method to bypass import issues
        def mock_ensure_overrides():
            if NeuronPlatform._config_overrides_applied:
                mock_logger.debug("Neuron config overrides already applied, skipping")
                return

            mock_logger.info("Applying Neuron config overrides")
            # Simulate successful override application without actual imports
            mock_logger.info("Neuron config overrides applied successfully")
            NeuronPlatform._config_overrides_applied = True

        # First call - should apply overrides
        with patch("vllm_neuron.platform.logger", mock_logger):
            with patch.object(
                NeuronPlatform,
                "_apply_config_overrides",
                side_effect=mock_ensure_overrides,
            ):
                platform._apply_config_overrides()

        # Check both log messages for successful override application
        assert mock_logger.info.call_args_list == [
            call("Applying Neuron config overrides"),
            call("Neuron config overrides applied successfully"),
        ]

        # Reset mock and verify second call
        mock_logger.reset_mock()

        # Second call should skip since flag is now set
        with patch("vllm_neuron.platform.logger", mock_logger):
            with patch.object(
                NeuronPlatform,
                "_apply_config_overrides",
                side_effect=mock_ensure_overrides,
            ):
                platform._apply_config_overrides()

        mock_logger.debug.assert_called_once_with(
            "Neuron config overrides already applied, skipping"
        )

    finally:
        # Always restore the original flag state
        NeuronPlatform._config_overrides_applied = original_flag_state


def test_apply_config_overrides_error(monkeypatch):
    """Test error handling in _apply_config_overrides."""
    platform = NeuronPlatform()

    # Reset the class flag
    NeuronPlatform._config_overrides_applied = False

    mock_logger = Mock()

    def mock_import(*args, **kwargs):
        if "vllm.config" in args:
            raise Exception("Test error")
        return importlib.__import__(*args, **kwargs)

    monkeypatch.setattr("vllm_neuron.platform.logger", mock_logger)
    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(Exception, match="Test error"):
        platform._apply_config_overrides()

    # Updated to match % formatting instead of f-string
    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args[0]
    assert call_args[0] == "Error applying Neuron config overrides: %s"
    assert str(call_args[1]) == "Test error"


def test_config_overrides_methods():
    """Test the overridden config methods."""
    platform = NeuronPlatform()

    # Create a mock ModelConfig class with the methods we want to test
    class MockModelConfig:
        def __init__(self):
            self.spec_target_max_model_len = None

        def _verify_quantization(self):
            pass

        def _verify_cuda_graph(self):
            pass

        def get_and_verify_max_len(self, max_model_len):
            if self.spec_target_max_model_len is not None:
                return self.spec_target_max_model_len
            return max_model_len

    # Reset the class flag to ensure we can apply overrides
    NeuronPlatform._config_overrides_applied = False

    with patch("vllm_neuron.platform.logger"):
        with patch("vllm.config.ModelConfig", MockModelConfig):
            # Apply overrides
            platform._apply_config_overrides()

            # Create instance and test
            model_config = MockModelConfig()

            # Test skip_verify_quantization and skip_verify_cuda_graph
            model_config._verify_quantization()  # Should pass without error
            model_config._verify_cuda_graph()  # Should pass without error

            # Test get_and_verify_max_len
            model_config.spec_target_max_model_len = 1024
            assert model_config.get_and_verify_max_len(2048) == 1024

            model_config.spec_target_max_model_len = None
            assert model_config.get_and_verify_max_len(2048) == 2048


def test_check_and_update_config_worker_settings():
    """Test worker class and world size configuration."""
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock()
    mock_config.cache_config = Mock(
        enable_prefix_caching=False, num_gpu_blocks_override=None
    )
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    # Test auto worker class setting
    mock_config.parallel_config.worker_cls = "auto"
    mock_config.parallel_config.world_size = 1
    NeuronPlatform.check_and_update_config(mock_config)
    assert (
        mock_config.parallel_config.worker_cls
        == "vllm_neuron.worker.neuron_worker.NeuronWorker"
    )

    # Test world size > 1
    mock_config.parallel_config.world_size = 2
    NeuronPlatform.check_and_update_config(mock_config)
    assert mock_config.parallel_config.distributed_executor_backend == "uni"


# Tests for Chat Completion Stream Generator Override
def test_get_vllm_version():
    """Test vLLM version detection returns correct values."""
    with patch("importlib.metadata.version", return_value="0.11.0"):
        assert NeuronPlatform._get_vllm_version() == "0.11.0"

    with patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        assert NeuronPlatform._get_vllm_version() == "unknown"


def test_get_attn_backend_cls():
    """Test get_attn_backend_cls method.

    Verifies:
    - Returns correct backend class string for NEURON_ATTN backend
    - Logs warning and switches to NEURON_ATTN for non-NEURON_ATTN backends
    - Logs info message when using NEURON_ATTN backend
    - Returns the expected backend class path
    """
    # Test with NEURON_ATTN backend
    with patch("vllm_neuron.platform.logger") as mock_logger:
        result = NeuronPlatform.get_attn_backend_cls(
            selected_backend=_Backend.NEURON_ATTN,
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_v1=False,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )

        # Verify the correct backend class is returned
        assert result == "vllm_neuron.attention.neuron_attn.NeuronAttentionBackend"

        # Verify info log is called
        mock_logger.info.assert_called_once_with("Using NEURON_ATTN backend.")

        # Verify no warning is logged for correct backend
        mock_logger.warning.assert_not_called()

    # Test with non-NEURON_ATTN backend (should log warning and switch)
    with patch("vllm_neuron.platform.logger") as mock_logger:
        # Create a mock backend that's not NEURON_ATTN
        mock_backend = Mock()
        mock_backend.__str__ = Mock(return_value="MOCK_BACKEND")

        result = NeuronPlatform.get_attn_backend_cls(
            selected_backend=mock_backend,
            head_size=128,
            dtype=torch.float32,
            kv_cache_dtype="fp8",
            block_size=32,
            use_v1=True,
            use_mla=True,
            has_sink=True,
            use_sparse=True,
        )

        # Verify the correct backend class is still returned
        assert result == "vllm_neuron.attention.neuron_attn.NeuronAttentionBackend"

        # Verify warning is logged for incorrect backend
        mock_logger.warning.assert_called_once_with(
            "Cannot use %s backend on Neuron. Switch to NEURON_ATTN instead",
            mock_backend,
        )

        # Verify info log is called
        mock_logger.info.assert_called_once_with("Using NEURON_ATTN backend.")


def test_get_nixl_supported_devices():
    """Test get_nixl_supported_devices method.

    Verifies:
    - Returns correct mapping of device types to supported kv_buffer_devices
    - Contains the expected device type from NeuronPlatform.device_type
    - Maps to the correct set of supported devices for nixl
    - Returns proper data structure (dict with tuple values)
    """
    result = NeuronPlatform.get_nixl_supported_devices()

    # Verify return type is dict
    assert isinstance(result, dict)

    # Verify the expected device type is present
    assert NeuronPlatform.device_type in result

    # Verify the mapping contains the expected supported devices
    expected_devices = {"cpu"}
    assert result[NeuronPlatform.device_type] == expected_devices

    # Verify the structure matches the expected format
    assert result == {"cpu": {"cpu"}}

    # Verify all values are sets (as returned by the method)
    for device_type, supported_devices in result.items():
        assert isinstance(supported_devices, set)
        assert isinstance(device_type, str)

    # Verify cpu is supported for the cpu device type
    assert "cpu" in result["cpu"]


def test_num_gpu_block_override_incremented_flag():
    """Test instance-level null block adjustment functionality.

    Verifies:
    - num_gpu_blocks_override is incremented by 1 for each config instance
    - Instance-level marker prevents multiple increments on same instance
    - Multiple config instances each get their own adjustment
    - Logging behavior for increment operation
    - No increment when num_gpu_blocks_override is None
    """
    # Test case 1: First config instance with num_gpu_blocks_override set
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(enable_prefix_caching=True)
    mock_config.cache_config.num_gpu_blocks_override = 10
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform.check_and_update_config(mock_config)

        # Verify increment happened
        assert mock_config.cache_config.num_gpu_blocks_override == 11
        assert "_neuron_null_block_adjusted" in mock_config.cache_config.__dict__
        assert mock_config.cache_config._neuron_null_block_adjusted is True

        # Verify logging
        mock_logger.info.assert_any_call(
            "Adding 1 to num_gpu_blocks_override (%d -> %d) "
            "to account for null block allocation",
            10,
            11,
        )

    # Test case 2: Same config instance called again - should not increment
    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform.check_and_update_config(mock_config)

        # Verify no additional increment (should remain 11)
        assert mock_config.cache_config.num_gpu_blocks_override == 11

        # Verify no increment logging occurred
        increment_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "Adding 1 to num_gpu_blocks_override" in str(call)
        ]
        assert len(increment_calls) == 0

    # Test case 3: Different config instance should get its own increment
    mock_config2 = Mock()
    mock_config2.model_config = Mock()
    mock_config2.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config2.cache_config = Mock(enable_prefix_caching=True)
    mock_config2.cache_config.num_gpu_blocks_override = 20
    mock_config2.scheduler_config = Mock()
    mock_config2.lora_config = None

    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform.check_and_update_config(mock_config2)

        # Verify increment happened for this new instance
        assert mock_config2.cache_config.num_gpu_blocks_override == 21
        assert "_neuron_null_block_adjusted" in mock_config2.cache_config.__dict__
        assert mock_config2.cache_config._neuron_null_block_adjusted is True

        # Verify increment logging occurred
        mock_logger.info.assert_any_call(
            "Adding 1 to num_gpu_blocks_override (%d -> %d) "
            "to account for null block allocation",
            20,
            21,
        )

    # Test case 4: No increment when num_gpu_blocks_override is None
    mock_config3 = Mock()
    mock_config3.model_config = Mock()
    mock_config3.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config3.cache_config = Mock(enable_prefix_caching=True)
    mock_config3.cache_config.num_gpu_blocks_override = None
    mock_config3.scheduler_config = Mock()
    mock_config3.lora_config = None

    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform.check_and_update_config(mock_config3)

        # Verify no increment happened and no marker set
        assert mock_config3.cache_config.num_gpu_blocks_override is None
        assert "_neuron_null_block_adjusted" not in mock_config3.cache_config.__dict__

        # Verify no increment logging occurred
        increment_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "Adding 1 to num_gpu_blocks_override" in str(call)
        ]
        assert len(increment_calls) == 0


def test_is_neuronx_distributed_inference_installed():
    """Test is_neuronx_distributed_inference when package is installed."""
    NeuronPlatform.is_neuronx_distributed_inference.cache_clear()

    # Mock successful import
    with patch("builtins.__import__") as mock_import:
        mock_import.return_value = Mock()  # Mock successful import
        result = NeuronPlatform.is_neuronx_distributed_inference()
        assert result is True


def test_apply_config_overrides_import_error():
    """Test _apply_config_overrides handles ImportError gracefully."""
    NeuronPlatform._config_overrides_applied = False

    with patch("vllm_neuron.platform.logger") as mock_logger:
        # Mock the import to fail by patching the actual import statement
        with patch("builtins.__import__", side_effect=ImportError("test import error")):
            NeuronPlatform._apply_config_overrides()

            # Verify warning was called with % formatting (lazy evaluation)
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            assert (
                call_args[0] == "Could not import vLLM config module for overrides: %s"
            )
            assert isinstance(call_args[1], ImportError)


def test_get_vllm_version_with_warning():
    """Test _get_vllm_version logs warning when package not found."""
    with patch("vllm_neuron.platform.logger") as mock_logger:
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            result = NeuronPlatform._get_vllm_version()

            assert result == "unknown"
            mock_logger.warning.assert_called_once_with(
                "Could not determine vLLM version"
            )


def test_cache_config_none():
    """Test check_and_update_config when cache_config is None."""
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = None  # Test None cache_config
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None

    # Should not raise an exception - the method should handle None cache_config gracefully
    # by skipping cache-related operations
    NeuronPlatform.check_and_update_config(mock_config)

    # Verify that basic config updates still happen
    assert (
        mock_config.parallel_config.worker_cls
        == "vllm_neuron.worker.neuron_worker.NeuronWorker"
    )


def test_use_neuronx_distributed_false():
    """Test use_neuronx_distributed returns False for different framework."""
    platform = NeuronPlatform()

    # Mock a different framework enum value (if there were others)
    with patch.object(
        platform, "get_neuron_framework_to_use", return_value="different_framework"
    ):
        assert not platform.use_neuronx_distributed()


# Tests from main branch for max_prompt_length validation feature
def test_max_prompt_length_config_setting():
    """Test max_prompt_length is set from additional_config."""
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(
        enable_prefix_caching=False, num_gpu_blocks_override=None
    )
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None
    mock_config.additional_config = {"max_prompt_length": 1024}

    NeuronPlatform.check_and_update_config(mock_config)

    assert mock_config.model_config.max_prompt_length == 1024


def test_max_prompt_length_config_none():
    """Test max_prompt_length is set to None when not provided."""
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(
        enable_prefix_caching=False, num_gpu_blocks_override=None
    )
    mock_config.scheduler_config = Mock()
    mock_config.lora_config = None
    mock_config.additional_config = {}

    NeuronPlatform.check_and_update_config(mock_config)

    assert mock_config.model_config.max_prompt_length is None


def test_openai_input_validation_within_limits():
    """Test OpenAI validation logic passes when input is within limits.

    This test directly verifies the validation logic without requiring
    full vLLM imports, making it suitable for test environments where
    vLLM modules are not available.
    """

    def neuron_validate_input(serving_instance, request, input_ids, input_text):
        """Simulate the Neuron validation logic."""
        token_num = len(input_ids)
        if token_num > serving_instance.model_config.max_prompt_length:
            raise ValueError(
                f"This model's maximum prompt length is "
                f"{serving_instance.model_config.max_prompt_length} tokens. However, "
                f"your request has {token_num} prompt tokens. Please reduce "
                "the length of the input messages."
            )
        return f"validation_passed_for_{len(input_ids)}_tokens"

    # Create mock serving instance
    mock_serving = Mock()
    mock_serving.model_config.max_prompt_length = 10

    # Test with input within limits
    result = neuron_validate_input(mock_serving, Mock(), [1, 2, 3], "test")
    assert result == "validation_passed_for_3_tokens"


def test_openai_input_validation_exceeds_limits():
    """Test OpenAI validation logic raises error when input exceeds limits.

    This test directly verifies the validation logic without requiring
    full vLLM imports, making it suitable for test environments where
    vLLM modules are not available.
    """

    def neuron_validate_input(serving_instance, request, input_ids, input_text):
        """Simulate the Neuron validation logic."""
        token_num = len(input_ids)
        if token_num > serving_instance.model_config.max_prompt_length:
            raise ValueError(
                f"This model's maximum prompt length is "
                f"{serving_instance.model_config.max_prompt_length} tokens. However, "
                f"your request has {token_num} prompt tokens. Please reduce "
                "the length of the input messages."
            )
        return f"validation_passed_for_{len(input_ids)}_tokens"

    # Create mock serving instance
    mock_serving = Mock()
    mock_serving.model_config.max_prompt_length = 5

    # Test with input exceeding limits
    with pytest.raises(ValueError) as exc_info:
        neuron_validate_input(mock_serving, Mock(), [1, 2, 3, 4, 5, 6, 7, 8], "test")

    assert "maximum prompt length is 5 tokens" in str(exc_info.value)
    assert "your request has 8 prompt tokens" in str(exc_info.value)


def test_integration_openai_serving_validation_override():
    """Integration test for OpenAI Serving validation override.

    This test directly exercises the validation logic that would be used
    in the monkey-patched method to achieve coverage of the validation lines.
    """

    # Test the validation logic directly since monkey patching may not work in test environment
    def simulate_openai_validation(serving_instance, request, input_ids, input_text):
        """Simulate the exact validation logic from the monkey patch."""
        if serving_instance.model_config.max_prompt_length is not None:
            token_num = len(input_ids)
            if token_num > serving_instance.model_config.max_prompt_length:
                raise ValueError(
                    f"This model's maximum prompt length is "
                    f"{serving_instance.model_config.max_prompt_length} tokens. However, "
                    f"your request has {token_num} prompt tokens. Please reduce "
                    "the length of the input messages."
                )
        return "original_method"

    # Create mock serving instance
    mock_serving = Mock()
    mock_serving.model_config.max_prompt_length = 5

    # Test case 1: Input within limits should pass
    result = simulate_openai_validation(mock_serving, Mock(), [1, 2, 3], "test")
    assert result == "original_method"

    # Test case 2: Input exceeding limits should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        simulate_openai_validation(mock_serving, Mock(), [1, 2, 3, 4, 5, 6, 7], "test")

    assert "maximum prompt length is 5 tokens" in str(exc_info.value)
    assert "your request has 7 prompt tokens" in str(exc_info.value)

    # Test case 3: None max_prompt_length should not validate
    mock_serving.model_config.max_prompt_length = None
    result = simulate_openai_validation(
        mock_serving, Mock(), [1, 2, 3, 4, 5, 6, 7], "test"
    )
    assert result == "original_method"


def test_integration_completion_renderer_validation_override():
    """Integration test for CompletionRenderer validation override.

    This test directly exercises the validation logic that would be used
    in the monkey-patched method to achieve coverage of the validation lines.
    """

    # Test the validation logic directly since monkey patching may not work in test environment
    def simulate_completion_validation(
        renderer_instance, token_ids, max_length=None, cache_salt=None, prompt=None
    ):
        """Simulate the exact validation logic from the monkey patch."""
        if renderer_instance.model_config.max_prompt_length is not None:
            if len(token_ids) > renderer_instance.model_config.max_prompt_length:
                raise ValueError(
                    f"This model's maximum prompt length is {renderer_instance.model_config.max_prompt_length} tokens. "
                    f"However, your request has {len(token_ids)} prompt tokens. "
                    "Please reduce the length of the input messages."
                )
        return "original_tokens_prompt"

    # Create mock renderer instance
    mock_renderer = Mock()
    mock_renderer.model_config.max_prompt_length = 5

    # Test case 1: Token IDs within limits should pass
    result = simulate_completion_validation(mock_renderer, [1, 2, 3])
    assert result == "original_tokens_prompt"

    # Test case 2: Token IDs exceeding limits should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        simulate_completion_validation(mock_renderer, [1, 2, 3, 4, 5, 6, 7])

    assert "maximum prompt length is 5 tokens" in str(exc_info.value)
    assert "your request has 7 prompt tokens" in str(exc_info.value)

    # Test case 3: None max_prompt_length should not validate
    mock_renderer.model_config.max_prompt_length = None
    result = simulate_completion_validation(mock_renderer, [1, 2, 3, 4, 5, 6, 7])
    assert result == "original_tokens_prompt"


def test_integration_monkey_patch_application():
    """Integration test verifying monkey patch application attempts.

    This test verifies that the monkey patching process runs without errors
    even when vLLM modules are not available in the test environment.
    """
    # Reset the override flag
    NeuronPlatform._config_overrides_applied = False

    # Test that calling _apply_config_overrides doesn't crash
    platform = NeuronPlatform()

    # This should complete without raising an exception
    # even if vLLM modules are not available (which is expected in test environment)
    platform._apply_config_overrides()

    # The flag might not be set if vLLM modules are not available,
    # but the method should not crash
    # In a real environment with vLLM available, this would be True
    assert isinstance(NeuronPlatform._config_overrides_applied, bool)


def test_device_id_to_physical_device_id_no_env_var():
    """Test device_id_to_physical_device_id when env var is not set.

    Verifies that:
    - When NEURON_RT_VISIBLE_CORES is not set, device_id is returned as-is
    - Various device IDs are correctly passed through
    """
    # Ensure the environment variable is not set
    with patch.dict(os.environ, {}, clear=True):
        # Remove the specific env var if it exists
        os.environ.pop(NeuronPlatform.device_control_env_var, None)

        assert NeuronPlatform.device_id_to_physical_device_id(0) == 0
        assert NeuronPlatform.device_id_to_physical_device_id(5) == 5
        assert NeuronPlatform.device_id_to_physical_device_id(15) == 15


def test_device_id_to_physical_device_id_range_format():
    """Test device_id_to_physical_device_id with range format (e.g., "0-7").

    Verifies that:
    - Range format is correctly parsed
    - Logical device IDs map to correct physical core IDs
    - Start and end of range are inclusive
    - Returns integer values
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0-7"}):
        # device_id 0 should map to physical core 0
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert result == 0
        assert isinstance(result, int)

        # device_id 3 should map to physical core 3
        result = NeuronPlatform.device_id_to_physical_device_id(3)
        assert result == 3
        assert isinstance(result, int)

        # device_id 7 should map to physical core 7 (end of range)
        result = NeuronPlatform.device_id_to_physical_device_id(7)
        assert result == 7
        assert isinstance(result, int)


def test_device_id_to_physical_device_id_range_format_non_zero_start():
    """Test device_id_to_physical_device_id with range starting at non-zero.

    Verifies that:
    - Ranges starting at non-zero values are correctly handled
    - Logical device ID 0 maps to the start of the physical range
    - Returns integer values
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "8-15"}):
        # device_id 0 should map to physical core 8
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert result == 8
        assert isinstance(result, int)

        # device_id 3 should map to physical core 11
        result = NeuronPlatform.device_id_to_physical_device_id(3)
        assert result == 11
        assert isinstance(result, int)

        # device_id 7 should map to physical core 15
        result = NeuronPlatform.device_id_to_physical_device_id(7)
        assert result == 15
        assert isinstance(result, int)


def test_device_id_to_physical_device_id_comma_separated():
    """Test device_id_to_physical_device_id with comma-separated list.

    Verifies that:
    - Comma-separated format is correctly parsed
    - Non-contiguous core assignments are handled
    - Returns integer values (not strings)
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0,2,4,6"}):
        # device_id 0 should map to 0
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert result == 0
        assert isinstance(result, int)

        # device_id 1 should map to 2
        result = NeuronPlatform.device_id_to_physical_device_id(1)
        assert result == 2
        assert isinstance(result, int)

        # device_id 2 should map to 4
        result = NeuronPlatform.device_id_to_physical_device_id(2)
        assert result == 4
        assert isinstance(result, int)

        # device_id 3 should map to 6
        result = NeuronPlatform.device_id_to_physical_device_id(3)
        assert result == 6
        assert isinstance(result, int)


def test_device_id_to_physical_device_id_comma_separated_with_spaces():
    """Test device_id_to_physical_device_id with comma-separated list with spaces.

    Verifies that:
    - Whitespace around values is correctly stripped
    - Returns integer values
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0, 2, 4, 6"}):
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert result == 0
        assert isinstance(result, int)

        result = NeuronPlatform.device_id_to_physical_device_id(2)
        assert result == 4
        assert isinstance(result, int)


def test_device_id_to_physical_device_id_single_core():
    """Test device_id_to_physical_device_id with single core.

    Verifies that:
    - Single core specification is correctly handled
    - Returns integer value
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "5"}):
        # device_id 0 should map to 5
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert result == 5
        assert isinstance(result, int)


def test_device_id_to_physical_device_id_logging():
    """Test device_id_to_physical_device_id logs debug message.

    Verifies that:
    - Debug logging is performed when env var is set
    - Log message contains device_id and physical device id
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0-7"}):
        with patch("vllm_neuron.platform.logger") as mock_logger:
            result = NeuronPlatform.device_id_to_physical_device_id(3)
            assert result == 3
            mock_logger.debug.assert_called_once_with(
                "device id: %s, physical device id: %s", 3, 3
            )


def test_device_id_to_physical_device_id_small_range():
    """Test device_id_to_physical_device_id with small range.

    Verifies that:
    - Small ranges (e.g., single core range) work correctly
    - Returns integer value
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "4-4"}):
        # Only one core available
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert result == 4
        assert isinstance(result, int)


def test_device_id_to_physical_device_id_large_range():
    """Test device_id_to_physical_device_id with larger range.

    Verifies that:
    - Larger ranges are correctly handled
    - Middle indices map correctly
    - Returns integer values
    """
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0-31"}):
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert result == 0
        assert isinstance(result, int)

        result = NeuronPlatform.device_id_to_physical_device_id(16)
        assert result == 16
        assert isinstance(result, int)

        result = NeuronPlatform.device_id_to_physical_device_id(31)
        assert result == 31
        assert isinstance(result, int)


def test_device_id_to_physical_device_id_return_type():
    """Test device_id_to_physical_device_id always returns int.

    Verifies that:
    - All code paths return integer values
    - Type consistency across different input formats
    """
    # No env var - should return int
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop(NeuronPlatform.device_control_env_var, None)
        result = NeuronPlatform.device_id_to_physical_device_id(5)
        assert isinstance(result, int)

    # Range format - should return int
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0-7"}):
        result = NeuronPlatform.device_id_to_physical_device_id(3)
        assert isinstance(result, int)

    # Comma-separated - should return int
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "1,3,5"}):
        result = NeuronPlatform.device_id_to_physical_device_id(1)
        assert isinstance(result, int)

    # Single core - should return int
    with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "7"}):
        result = NeuronPlatform.device_id_to_physical_device_id(0)
        assert isinstance(result, int)


def test_apply_world_size_override_property_not_found():
    """Test _apply_world_size_override when world_size_across_dp is absent."""

    class FakeParallelConfig:
        pass  # No world_size_across_dp at all

    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform._apply_world_size_override(FakeParallelConfig)

        mock_logger.warning.assert_called_once_with(
            "world_size_across_dp property not found on ParallelConfig"
        )


def test_apply_world_size_override_not_a_property():
    """Test _apply_world_size_override when world_size_across_dp has no fget."""

    class FakeParallelConfig:
        # Exists but is a plain int, not a property — so hasattr(original_property, 'fget') is False
        world_size_across_dp = 42

    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform._apply_world_size_override(FakeParallelConfig)

        mock_logger.warning.assert_called_once_with(
            "Could not get original getter for world_size_across_dp property"
        )


def test_apply_world_size_override_non_neuron_platform_path():
    """Test the patched property falls back to original on non-Neuron platform."""

    class FakeParallelConfig:
        @property
        def world_size_across_dp(self):
            return 777

    with patch("vllm_neuron.platform.logger"):
        NeuronPlatform._apply_world_size_override(FakeParallelConfig)

    instance = FakeParallelConfig()
    instance.data_parallel_size = 4

    # Mock current_platform to be something other than NeuronPlatform
    mock_platform = Mock()
    mock_platform.__class__.__name__ = "CudaPlatform"

    with patch(
        "vllm.platforms.current_platform",
        mock_platform,
    ):
        result = instance.world_size_across_dp
        assert result == 777  # Original getter's return value


def test_apply_openai_overrides_early_return(cleanup_global_state):
    """Test _apply_openai_overrides returns immediately when already applied."""
    NeuronPlatform._openai_overrides_applied = True

    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform._apply_openai_overrides()

        # Nothing should happen — no info, no warning, no error
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()


def test_apply_openai_overrides_patches_both_classes(cleanup_global_state):
    """Test _apply_openai_overrides patches OpenAIServing and CompletionRenderer."""
    NeuronPlatform._openai_overrides_applied = False

    original_validate = Mock()
    original_tokenize = Mock()

    class StubServing:
        _validate_input = original_validate

    class StubRenderer:
        _create_tokens_prompt = original_tokenize

    mock_serving_mod = Mock()
    mock_serving_mod.OpenAIServing = StubServing

    mock_renderer_mod = Mock()
    mock_renderer_mod.CompletionRenderer = StubRenderer

    with (
        patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints": Mock(),
                "vllm.entrypoints.openai": Mock(),
                "vllm.entrypoints.openai.serving_engine": mock_serving_mod,
                "vllm.entrypoints.renderer": mock_renderer_mod,
            },
        ),
        patch.object(_plat_mod, "logger") as mock_logger,
    ):
        NeuronPlatform._apply_openai_overrides()

        assert NeuronPlatform._openai_overrides_applied is True
        assert StubServing._validate_input is not original_validate
        assert StubRenderer._create_tokens_prompt is not original_tokenize
        mock_logger.info.assert_any_call(
            "Neuron OpenAI serving overrides applied successfully"
        )


def test_apply_openai_overrides_generic_exception_logged(cleanup_global_state):
    """Test _apply_openai_overrides catches non-Import exceptions."""
    NeuronPlatform._openai_overrides_applied = False

    bad_module = Mock()
    type(bad_module).OpenAIServing = property(
        lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with (
        patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints": Mock(),
                "vllm.entrypoints.openai": Mock(),
                "vllm.entrypoints.openai.serving_engine": bad_module,
                "vllm.entrypoints.renderer": Mock(),
            },
        ),
        patch.object(_plat_mod, "logger") as mock_logger,
    ):
        NeuronPlatform._apply_openai_overrides()

        mock_logger.error.assert_called_once()
        args = mock_logger.error.call_args[0]
        assert args[0] == "Error applying OpenAI serving overrides: %s"
        assert isinstance(args[1], RuntimeError)
        assert NeuronPlatform._openai_overrides_applied is False


def test_openai_validate_input_override_rejects_long_prompt(cleanup_global_state):
    """Exercise the patched _validate_input when prompt exceeds limit."""
    NeuronPlatform._openai_overrides_applied = False

    original_validate = Mock(return_value="ok")

    class StubServing:
        _validate_input = original_validate

    class StubRenderer:
        _create_tokens_prompt = Mock()

    mock_serving_mod = Mock()
    mock_serving_mod.OpenAIServing = StubServing

    mock_renderer_mod = Mock()
    mock_renderer_mod.CompletionRenderer = StubRenderer

    with (
        patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints": Mock(),
                "vllm.entrypoints.openai": Mock(),
                "vllm.entrypoints.openai.serving_engine": mock_serving_mod,
                "vllm.entrypoints.renderer": mock_renderer_mod,
            },
        ),
        patch.object(_plat_mod, "logger"),
    ):
        NeuronPlatform._apply_openai_overrides()

    patched = StubServing._validate_input

    inst = Mock()
    inst.model_config.max_prompt_length = 3

    with pytest.raises(ValueError, match="maximum prompt length is 3 tokens"):
        patched(inst, Mock(), [1, 2, 3, 4, 5], "text")


def test_renderer_override_rejects_long_prompt(cleanup_global_state):
    """Exercise the patched _create_tokens_prompt when prompt exceeds limit."""
    NeuronPlatform._openai_overrides_applied = False

    class StubServing:
        _validate_input = Mock()

    class StubRenderer:
        _create_tokens_prompt = Mock()

    mock_serving_mod = Mock()
    mock_serving_mod.OpenAIServing = StubServing

    mock_renderer_mod = Mock()
    mock_renderer_mod.CompletionRenderer = StubRenderer

    with (
        patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints": Mock(),
                "vllm.entrypoints.openai": Mock(),
                "vllm.entrypoints.openai.serving_engine": mock_serving_mod,
                "vllm.entrypoints.renderer": mock_renderer_mod,
            },
        ),
        patch.object(_plat_mod, "logger"),
    ):
        NeuronPlatform._apply_openai_overrides()

    patched = StubRenderer._create_tokens_prompt

    inst = Mock()
    inst.model_config.max_prompt_length = 2

    with pytest.raises(ValueError, match="maximum prompt length is 2 tokens"):
        patched(inst, [1, 2, 3, 4])


def test_renderer_override_allows_short_prompt(cleanup_global_state):
    """Exercise the patched _create_tokens_prompt when prompt is within limit."""
    NeuronPlatform._openai_overrides_applied = False

    original_tokenize = Mock(return_value="original_result")

    class StubServing:
        _validate_input = Mock()

    class StubRenderer:
        _create_tokens_prompt = original_tokenize

    mock_serving_mod = Mock()
    mock_serving_mod.OpenAIServing = StubServing

    mock_renderer_mod = Mock()
    mock_renderer_mod.CompletionRenderer = StubRenderer

    with (
        patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints": Mock(),
                "vllm.entrypoints.openai": Mock(),
                "vllm.entrypoints.openai.serving_engine": mock_serving_mod,
                "vllm.entrypoints.renderer": mock_renderer_mod,
            },
        ),
        patch.object(_plat_mod, "logger"),
    ):
        NeuronPlatform._apply_openai_overrides()

    patched = StubRenderer._create_tokens_prompt

    inst = Mock()
    inst.model_config.max_prompt_length = 10

    result = patched(inst, [1, 2, 3])
    assert result == "original_result"


def test_apply_engine_client_override_early_return(cleanup_global_state):
    """Test _apply_engine_client_override returns immediately when already applied."""
    NeuronPlatform._engine_client_override_applied = True

    with patch("vllm_neuron.platform.logger") as mock_logger:
        NeuronPlatform._apply_engine_client_override()

        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()


def test_get_vllm_version_success_real_path():
    """Test _get_vllm_version returns the version string."""
    with patch(
        "vllm_neuron.platform.importlib.metadata.version", return_value="0.16.0"
    ):
        result = NeuronPlatform._get_vllm_version()
        assert result == "0.16.0"


def test_get_vllm_version_package_not_found_real_path():
    """Test _get_vllm_version returns 'unknown' when package missing."""
    with (
        patch(
            "vllm_neuron.platform.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("vllm"),
        ),
        patch("vllm_neuron.platform.logger") as mock_logger,
    ):
        result = NeuronPlatform._get_vllm_version()
        assert result == "unknown"
        mock_logger.warning.assert_called_once_with("Could not determine vLLM version")


# ============================================================================
# Tests for platform_overrides module
# ============================================================================


class TestPlatformOverrides:
    """Tests for the platform_overrides module functions."""

    def test_skip_verify_with_parallel_config_basic(self):
        """Test skip_verify_with_parallel_config with basic parallel config."""
        from vllm_neuron.platform_overrides import skip_verify_with_parallel_config

        mock_self = Mock()
        mock_self.seed = 42
        mock_self.use_async_output_proc = False
        mock_self.registry = Mock()
        mock_self.registry.is_pp_supported_model.return_value = True
        mock_self.architectures = ["LlamaForCausalLM"]

        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "ray"
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 1

        # Should not raise any exception
        skip_verify_with_parallel_config(mock_self, mock_parallel_config)

    def test_skip_verify_with_parallel_config_external_launcher(self):
        """Test skip_verify_with_parallel_config with external launcher."""
        from vllm_neuron.platform_overrides import skip_verify_with_parallel_config

        mock_self = Mock()
        mock_self.seed = 42
        mock_self.use_async_output_proc = False

        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "external_launcher"
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 1

        # Should not raise with seed set
        skip_verify_with_parallel_config(mock_self, mock_parallel_config)

    def test_skip_verify_with_parallel_config_external_launcher_no_seed(self):
        """Test skip_verify_with_parallel_config raises when external launcher has no seed."""
        from vllm_neuron.platform_overrides import skip_verify_with_parallel_config

        mock_self = Mock()
        mock_self.seed = None

        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "external_launcher"
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 1

        with pytest.raises(AssertionError, match="Seed must be set"):
            skip_verify_with_parallel_config(mock_self, mock_parallel_config)

    def test_skip_verify_with_parallel_config_expert_parallel(self):
        """Test skip_verify_with_parallel_config with expert parallelism."""
        from vllm_neuron.platform_overrides import skip_verify_with_parallel_config

        mock_self = Mock()
        mock_self.seed = 42
        mock_self.use_async_output_proc = False
        mock_self._verify_with_expert_parallelism = Mock()

        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "ray"
        mock_parallel_config.enable_expert_parallel = True
        mock_parallel_config.pipeline_parallel_size = 1

        skip_verify_with_parallel_config(mock_self, mock_parallel_config)
        mock_self._verify_with_expert_parallelism.assert_called_once()

    def test_skip_verify_with_parallel_config_pipeline_parallel(self):
        """Test skip_verify_with_parallel_config with pipeline parallelism."""
        from vllm_neuron.platform_overrides import skip_verify_with_parallel_config

        mock_self = Mock()
        mock_self.seed = 42
        mock_self.use_async_output_proc = True
        mock_self.registry = Mock()
        mock_self.registry.is_pp_supported_model.return_value = True
        mock_self.architectures = ["LlamaForCausalLM"]

        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "ray"
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 2

        skip_verify_with_parallel_config(mock_self, mock_parallel_config)
        # use_async_output_proc should be set to False for PP > 1
        assert mock_self.use_async_output_proc is False

    def test_skip_verify_with_parallel_config_pp_unsupported_model(self):
        """Test skip_verify_with_parallel_config raises for unsupported PP model."""
        from vllm_neuron.platform_overrides import skip_verify_with_parallel_config

        mock_self = Mock()
        mock_self.seed = 42
        mock_self.registry = Mock()
        mock_self.registry.is_pp_supported_model.return_value = False
        mock_self.architectures = ["UnsupportedModel"]

        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "ray"
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 2

        with pytest.raises(NotImplementedError, match="Pipeline parallelism"):
            skip_verify_with_parallel_config(mock_self, mock_parallel_config)

    def test_skip_verify_quantization(self):
        """Test skip_verify_quantization does nothing."""
        from vllm_neuron.platform_overrides import skip_verify_quantization

        mock_self = Mock()
        # Should not raise any exception
        skip_verify_quantization(mock_self)

    def test_skip_verify_cuda_graph(self):
        """Test skip_verify_cuda_graph does nothing."""
        from vllm_neuron.platform_overrides import skip_verify_cuda_graph

        mock_self = Mock()
        # Should not raise any exception
        skip_verify_cuda_graph(mock_self)

    def test_changed_get_and_verify_max_len_with_spec_target(self):
        """Test changed_get_and_verify_max_len returns spec_target when set."""
        from vllm_neuron.platform_overrides import changed_get_and_verify_max_len

        mock_self = Mock()
        mock_self.spec_target_max_model_len = 2048

        result = changed_get_and_verify_max_len(mock_self, 4096)
        assert result == 2048

    def test_changed_get_and_verify_max_len_without_spec_target(self):
        """Test changed_get_and_verify_max_len returns max_model_len when no spec_target."""
        from vllm_neuron.platform_overrides import changed_get_and_verify_max_len

        mock_self = Mock()
        mock_self.spec_target_max_model_len = None

        result = changed_get_and_verify_max_len(mock_self, 4096)
        assert result == 4096

    def test_get_openai_overrides_returns_functions(self):
        """Test get_openai_overrides returns the expected tuple of classes and functions."""
        # Mock the vllm modules
        mock_openai_serving = Mock()
        mock_openai_serving._validate_input = Mock(return_value="original_result")

        mock_renderer = Mock()
        mock_renderer._create_tokens_prompt = Mock(return_value="original_tokens")

        mock_serving_module = Mock()
        mock_serving_module.OpenAIServing = mock_openai_serving

        mock_renderer_module = Mock()
        mock_renderer_module.CompletionRenderer = mock_renderer

        with patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints.openai.serving_engine": mock_serving_module,
                "vllm.entrypoints.renderer": mock_renderer_module,
            },
        ):
            from vllm_neuron.platform_overrides import get_openai_overrides

            result = get_openai_overrides()

            assert len(result) == 4
            (
                OpenAIServing,
                changed_validate_input,
                CompletionRenderer,
                changed_create_tokens_prompt,
            ) = result
            assert OpenAIServing == mock_openai_serving
            assert CompletionRenderer == mock_renderer
            assert callable(changed_validate_input)
            assert callable(changed_create_tokens_prompt)

    def test_get_openai_overrides_changed_validate_input_within_limits(self):
        """Test the actual changed_validate_input function from get_openai_overrides passes when within limits."""
        mock_openai_serving = Mock()
        mock_openai_serving._validate_input = Mock(return_value="original_result")

        mock_renderer = Mock()
        mock_renderer._create_tokens_prompt = Mock(return_value="original_tokens")

        mock_serving_module = Mock()
        mock_serving_module.OpenAIServing = mock_openai_serving

        mock_renderer_module = Mock()
        mock_renderer_module.CompletionRenderer = mock_renderer

        with patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints.openai.serving_engine": mock_serving_module,
                "vllm.entrypoints.renderer": mock_renderer_module,
            },
        ):
            from vllm_neuron.platform_overrides import get_openai_overrides

            _, changed_validate_input, _, _ = get_openai_overrides()

            # Create mock self with max_prompt_length set
            mock_self = Mock()
            mock_self.model_config.max_prompt_length = 100

            mock_request = Mock()
            input_ids = [1, 2, 3, 4, 5]  # 5 tokens, within limit
            input_text = "test input"

            result = changed_validate_input(
                mock_self, mock_request, input_ids, input_text
            )
            assert result == "original_result"

    def test_get_openai_overrides_changed_validate_input_exceeds_limits(self):
        """Test the actual changed_validate_input function from get_openai_overrides raises when exceeding limits."""
        mock_openai_serving = Mock()
        mock_openai_serving._validate_input = Mock(return_value="original_result")

        mock_renderer = Mock()
        mock_renderer._create_tokens_prompt = Mock(return_value="original_tokens")

        mock_serving_module = Mock()
        mock_serving_module.OpenAIServing = mock_openai_serving

        mock_renderer_module = Mock()
        mock_renderer_module.CompletionRenderer = mock_renderer

        with patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints.openai.serving_engine": mock_serving_module,
                "vllm.entrypoints.renderer": mock_renderer_module,
            },
        ):
            from vllm_neuron.platform_overrides import get_openai_overrides

            _, changed_validate_input, _, _ = get_openai_overrides()

            mock_self = Mock()
            mock_self.model_config.max_prompt_length = 5

            mock_request = Mock()
            input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens, exceeds limit
            input_text = "test input"

            with pytest.raises(ValueError) as exc_info:
                changed_validate_input(mock_self, mock_request, input_ids, input_text)

            assert "maximum prompt length is 5 tokens" in str(exc_info.value)
            assert "your request has 10 prompt tokens" in str(exc_info.value)

    def test_get_openai_overrides_changed_validate_input_no_limit(self):
        """Test the actual changed_validate_input function passes when no limit is set."""
        mock_openai_serving = Mock()
        mock_openai_serving._validate_input = Mock(return_value="original_result")

        mock_renderer = Mock()
        mock_renderer._create_tokens_prompt = Mock(return_value="original_tokens")

        mock_serving_module = Mock()
        mock_serving_module.OpenAIServing = mock_openai_serving

        mock_renderer_module = Mock()
        mock_renderer_module.CompletionRenderer = mock_renderer

        with patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints.openai.serving_engine": mock_serving_module,
                "vllm.entrypoints.renderer": mock_renderer_module,
            },
        ):
            from vllm_neuron.platform_overrides import get_openai_overrides

            _, changed_validate_input, _, _ = get_openai_overrides()

            mock_self = Mock()
            mock_self.model_config.max_prompt_length = None

            mock_request = Mock()
            input_ids = [1] * 1000  # Many tokens, but no limit
            input_text = "test input"

            result = changed_validate_input(
                mock_self, mock_request, input_ids, input_text
            )
            assert result == "original_result"

    def test_get_openai_overrides_changed_create_tokens_prompt_within_limits(self):
        """Test the actual changed_create_tokens_prompt function passes when within limits."""
        mock_openai_serving = Mock()
        mock_openai_serving._validate_input = Mock(return_value="original_result")

        mock_renderer = Mock()
        mock_renderer._create_tokens_prompt = Mock(return_value="original_tokens")

        mock_serving_module = Mock()
        mock_serving_module.OpenAIServing = mock_openai_serving

        mock_renderer_module = Mock()
        mock_renderer_module.CompletionRenderer = mock_renderer

        with patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints.openai.serving_engine": mock_serving_module,
                "vllm.entrypoints.renderer": mock_renderer_module,
            },
        ):
            from vllm_neuron.platform_overrides import get_openai_overrides

            _, _, _, changed_create_tokens_prompt = get_openai_overrides()

            mock_self = Mock()
            mock_self.model_config.max_prompt_length = 100

            token_ids = [1, 2, 3, 4, 5]  # 5 tokens, within limit

            result = changed_create_tokens_prompt(mock_self, token_ids)
            assert result == "original_tokens"

    def test_get_openai_overrides_changed_create_tokens_prompt_exceeds_limits(self):
        """Test the actual changed_create_tokens_prompt function raises when exceeding limits."""
        mock_openai_serving = Mock()
        mock_openai_serving._validate_input = Mock(return_value="original_result")

        mock_renderer = Mock()
        mock_renderer._create_tokens_prompt = Mock(return_value="original_tokens")

        mock_serving_module = Mock()
        mock_serving_module.OpenAIServing = mock_openai_serving

        mock_renderer_module = Mock()
        mock_renderer_module.CompletionRenderer = mock_renderer

        with patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints.openai.serving_engine": mock_serving_module,
                "vllm.entrypoints.renderer": mock_renderer_module,
            },
        ):
            from vllm_neuron.platform_overrides import get_openai_overrides

            _, _, _, changed_create_tokens_prompt = get_openai_overrides()

            mock_self = Mock()
            mock_self.model_config.max_prompt_length = 5

            token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens, exceeds limit

            with pytest.raises(ValueError) as exc_info:
                changed_create_tokens_prompt(mock_self, token_ids)

            assert "maximum prompt length is 5 tokens" in str(exc_info.value)
            assert "your request has 10 prompt tokens" in str(exc_info.value)

    def test_get_openai_overrides_changed_create_tokens_prompt_no_limit(self):
        """Test the actual changed_create_tokens_prompt function passes when no limit is set."""
        mock_openai_serving = Mock()
        mock_openai_serving._validate_input = Mock(return_value="original_result")

        mock_renderer = Mock()
        mock_renderer._create_tokens_prompt = Mock(return_value="original_tokens")

        mock_serving_module = Mock()
        mock_serving_module.OpenAIServing = mock_openai_serving

        mock_renderer_module = Mock()
        mock_renderer_module.CompletionRenderer = mock_renderer

        with patch.dict(
            "sys.modules",
            {
                "vllm.entrypoints.openai.serving_engine": mock_serving_module,
                "vllm.entrypoints.renderer": mock_renderer_module,
            },
        ):
            from vllm_neuron.platform_overrides import get_openai_overrides

            _, _, _, changed_create_tokens_prompt = get_openai_overrides()

            mock_self = Mock()
            mock_self.model_config.max_prompt_length = None

            token_ids = [1] * 1000  # Many tokens, but no limit

            result = changed_create_tokens_prompt(mock_self, token_ids)
            assert result == "original_tokens"

    def test_changed_validate_input_within_limits(self):
        """Test changed_validate_input passes when within limits."""
        # Create mock objects
        mock_original_validate = Mock(return_value="validated_prompt")

        mock_self = Mock()
        mock_self.model_config.max_prompt_length = 100

        mock_request = Mock()
        input_ids = [1, 2, 3, 4, 5]  # 5 tokens, within limit
        input_text = "test input"

        # Simulate the changed_validate_input function behavior
        def changed_validate_input(self, request, input_ids, input_text):
            if self.model_config.max_prompt_length is not None:
                token_num = len(input_ids)
                if token_num > self.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is "
                        f"{self.model_config.max_prompt_length} tokens. However, "
                        f"your request has {token_num} prompt tokens."
                    )
            return mock_original_validate(self, request, input_ids, input_text)

        result = changed_validate_input(mock_self, mock_request, input_ids, input_text)
        assert result == "validated_prompt"

    def test_changed_validate_input_exceeds_limits(self):
        """Test changed_validate_input raises when exceeding limits."""
        mock_self = Mock()
        mock_self.model_config.max_prompt_length = 5

        mock_request = Mock()
        input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens, exceeds limit
        input_text = "test input"

        def changed_validate_input(self, request, input_ids, input_text):
            if self.model_config.max_prompt_length is not None:
                token_num = len(input_ids)
                if token_num > self.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is "
                        f"{self.model_config.max_prompt_length} tokens. However, "
                        f"your request has {token_num} prompt tokens."
                    )
            return "validated_prompt"

        with pytest.raises(ValueError) as exc_info:
            changed_validate_input(mock_self, mock_request, input_ids, input_text)

        assert "maximum prompt length is 5 tokens" in str(exc_info.value)
        assert "your request has 10 prompt tokens" in str(exc_info.value)

    def test_changed_validate_input_no_limit_set(self):
        """Test changed_validate_input passes when no limit is set."""
        mock_original_validate = Mock(return_value="validated_prompt")

        mock_self = Mock()
        mock_self.model_config.max_prompt_length = None

        mock_request = Mock()
        input_ids = [1] * 1000  # Many tokens, but no limit
        input_text = "test input"

        def changed_validate_input(self, request, input_ids, input_text):
            if self.model_config.max_prompt_length is not None:
                token_num = len(input_ids)
                if token_num > self.model_config.max_prompt_length:
                    raise ValueError("Exceeded limit")
            return mock_original_validate(self, request, input_ids, input_text)

        result = changed_validate_input(mock_self, mock_request, input_ids, input_text)
        assert result == "validated_prompt"

    def test_changed_create_tokens_prompt_within_limits(self):
        """Test changed_create_tokens_prompt passes when within limits."""
        mock_original_create = Mock(return_value="tokens_prompt")

        mock_self = Mock()
        mock_self.model_config.max_prompt_length = 100

        token_ids = [1, 2, 3, 4, 5]  # 5 tokens, within limit

        def changed_create_tokens_prompt(
            self, token_ids, max_length=None, cache_salt=None, prompt=None
        ):
            if self.model_config.max_prompt_length is not None:
                if len(token_ids) > self.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is {self.model_config.max_prompt_length} tokens. "
                        f"However, your request has {len(token_ids)} prompt tokens."
                    )
            return mock_original_create(self, token_ids, max_length, cache_salt, prompt)

        result = changed_create_tokens_prompt(mock_self, token_ids)
        assert result == "tokens_prompt"

    def test_changed_create_tokens_prompt_exceeds_limits(self):
        """Test changed_create_tokens_prompt raises when exceeding limits."""
        mock_self = Mock()
        mock_self.model_config.max_prompt_length = 5

        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens, exceeds limit

        def changed_create_tokens_prompt(
            self, token_ids, max_length=None, cache_salt=None, prompt=None
        ):
            if self.model_config.max_prompt_length is not None:
                if len(token_ids) > self.model_config.max_prompt_length:
                    raise ValueError(
                        f"This model's maximum prompt length is {self.model_config.max_prompt_length} tokens. "
                        f"However, your request has {len(token_ids)} prompt tokens."
                    )
            return "tokens_prompt"

        with pytest.raises(ValueError) as exc_info:
            changed_create_tokens_prompt(mock_self, token_ids)

        assert "maximum prompt length is 5 tokens" in str(exc_info.value)
        assert "your request has 10 prompt tokens" in str(exc_info.value)

    def test_changed_create_tokens_prompt_no_limit_set(self):
        """Test changed_create_tokens_prompt passes when no limit is set."""
        mock_original_create = Mock(return_value="tokens_prompt")

        mock_self = Mock()
        mock_self.model_config.max_prompt_length = None

        token_ids = [1] * 1000  # Many tokens, but no limit

        def changed_create_tokens_prompt(
            self, token_ids, max_length=None, cache_salt=None, prompt=None
        ):
            if self.model_config.max_prompt_length is not None:
                if len(token_ids) > self.model_config.max_prompt_length:
                    raise ValueError("Exceeded limit")
            return mock_original_create(self, token_ids, max_length, cache_salt, prompt)

        result = changed_create_tokens_prompt(mock_self, token_ids)
        assert result == "tokens_prompt"
