"""
Collision detection and surface sampling modules.

Includes:
- NAxisSampler: Surface sampling for multi-axis machining
"""

from pycam3d.collision.sampler_4axis import (
    NAxisSampler,
    SamplingResult,
)

__all__ = [
    "NAxisSampler",
    "SamplingResult",
]
