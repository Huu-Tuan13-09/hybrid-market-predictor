"""
backend/agents/__init__.py
LangGraph Multi-Agent package.

Exports:
  - AgentState  : TypedDict shared across all nodes
  - build_graph : Factory function returning compiled LangGraph app
"""

from .state import AgentState
from .graph import build_graph

__all__ = ["AgentState", "build_graph"]
