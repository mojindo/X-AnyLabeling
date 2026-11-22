"""Duplicate groups: exact copies become one node, then maximal cliques.

Near-duplicate is not transitive, so two cliques that only share a vertex
stay separate. Pixel-identical images are one supernode before that step,
so an exact copy inherits the near-duplicate edges of its twin.
"""
from __future__ import annotations

from collections import defaultdict


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _maximal_cliques(adj: dict[int, set[int]]) -> list[list[int]]:
    if not adj:
        return []
    cliques: list[list[int]] = []

    def bronk(rp: set[int], xp: set[int], sp: set[int]) -> None:
        if not xp and not sp:
            if len(rp) >= 2:
                cliques.append(sorted(rp))
            return
        pivot_set = xp | sp
        pivot = max(pivot_set, key=lambda u: len(adj[u] & sp))
        for v in list(sp - adj[pivot]):
            bronk(rp | {v}, xp & adj[v], sp & adj[v])
            xp.add(v)
            sp.discard(v)

    bronk(set(), set(), set(adj))
    uniq = {tuple(c): c for c in cliques}
    final: list[list[int]] = []
    for c in sorted(uniq.values(), key=lambda g: -len(g)):
        cs = set(c)
        if not any(cs < set(bigger) for bigger in final):
            final.append(c)
    return final


def build_dedup_groups(
    n: int,
    exact_groups: list[list[int]],
    delete_pairs: list[tuple[int, int]],
) -> list[list[int]]:
    """Return groups of size >= 2. Cliques are not merged with each other."""
    uf = UnionFind(n)
    for group in exact_groups:
        if len(group) < 2:
            continue
        head = group[0]
        for other in group[1:]:
            uf.union(head, other)

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[uf.find(i)].append(i)

    node_of = {i: uf.find(i) for i in range(n)}
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in delete_pairs:
        na, nb = node_of[a], node_of[b]
        if na == nb:
            continue
        adj[na].add(nb)
        adj[nb].add(na)

    groups: list[list[int]] = []
    for root, idxs in members.items():
        if len(idxs) < 2:
            continue
        if root not in adj:
            groups.append(sorted(idxs))

    for clique in _maximal_cliques(adj):
        expanded: list[int] = []
        for node in clique:
            expanded.extend(members[node])
        expanded = sorted(set(expanded))
        if len(expanded) >= 2:
            groups.append(expanded)

    groups.sort(key=lambda g: (-len(g), g[0] if g else 0))
    return groups