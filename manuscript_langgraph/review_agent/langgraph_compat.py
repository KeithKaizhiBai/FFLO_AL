from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

try:  # Production path when LangGraph is installed.
    from langgraph.graph import END, START, StateGraph  # type: ignore
except ImportError:  # Offline validation path; not advertised as LangGraph itself.
    START = "__start__"
    END = "__end__"

    class _CompiledGraph:
        def __init__(
            self,
            nodes: dict[str, Callable],
            edges: dict[str, list[str]],
            conditional_edges: dict[str, tuple[Callable, dict[str, str]]],
        ):
            self._nodes = nodes
            self._edges = edges
            self._conditional_edges = conditional_edges

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            current = self._edges[START][0]
            result = dict(state)
            visited: set[str] = set()
            while current != END:
                if current in visited:
                    raise RuntimeError(f"cycle detected at {current}")
                visited.add(current)
                update = self._nodes[current](result) or {}
                result.update(update)
                if current in self._conditional_edges:
                    router, path_map = self._conditional_edges[current]
                    route = router(result)
                    if route not in path_map:
                        raise RuntimeError(f"unknown conditional route {route!r} at {current}")
                    current = path_map[route]
                else:
                    next_nodes = self._edges.get(current, [END])
                    if len(next_nodes) != 1:
                        raise RuntimeError("offline compatibility runner requires one unconditional successor")
                    current = next_nodes[0]
            return result

    class StateGraph:
        def __init__(self, _state_type: object):
            self._nodes: dict[str, Callable] = {}
            self._edges: dict[str, list[str]] = defaultdict(list)
            self._conditional_edges: dict[str, tuple[Callable, dict[str, str]]] = {}

        def add_node(self, name: str, function: Callable) -> None:
            self._nodes[name] = function

        def add_edge(self, left: str, right: str) -> None:
            self._edges[left].append(right)

        def add_conditional_edges(
            self,
            source: str,
            path: Callable,
            path_map: dict[str, str],
        ) -> None:
            self._conditional_edges[source] = (path, path_map)

        def compile(self) -> _CompiledGraph:
            if START not in self._edges:
                raise RuntimeError("graph has no START edge")
            return _CompiledGraph(self._nodes, dict(self._edges), self._conditional_edges)
