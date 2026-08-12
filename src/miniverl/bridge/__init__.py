"""Pinned, fail-closed interoperability with one documented verl profile.

The bridge exchanges standard artifacts. It is deliberately not a distributed
runtime adapter and does not claim generic verl configuration compatibility.
"""

from miniverl.bridge.contract import (
    BRIDGE_PROFILE,
    VERL_COMMIT,
    VERL_REPOSITORY,
    VERL_TAG,
)
from miniverl.bridge.opd_v08 import VERL_OPD_V08_PROFILE

__all__ = [
    "BRIDGE_PROFILE",
    "VERL_COMMIT",
    "VERL_OPD_V08_PROFILE",
    "VERL_REPOSITORY",
    "VERL_TAG",
]
