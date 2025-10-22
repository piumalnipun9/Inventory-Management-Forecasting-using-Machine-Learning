"""Inventory analytics package.

Note: Avoid importing heavy submodules at package import time to keep
lightweight tools (like CSV converters) fast and dependency-light.
Import submodules directly where needed, e.g.:
	from src.adapters.grocery_csv_adapter import convert_grocery_csv_to_internal
"""

__all__ = []

