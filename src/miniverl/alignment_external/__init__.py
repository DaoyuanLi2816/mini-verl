"""Optional external alignment benchmark adapters.

Nothing here is imported by the torch-free core path. The evaluators live
behind the ``alignment-benchmarks`` extra so a plain ``pip install miniverl``
never pulls in datasets, judge models or network clients.
"""

from __future__ import annotations

__all__: list[str] = []
