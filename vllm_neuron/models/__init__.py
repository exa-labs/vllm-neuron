# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree model registrations for the Neuron plugin."""

from vllm_neuron.models.nemotron_embed import register_nemotron_embed

__all__ = ["register_nemotron_embed"]
