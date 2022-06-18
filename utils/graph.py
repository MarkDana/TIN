#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy
from itertools import chain, combinations

class Node(object):
    def __init__(self):
        self.To = set()
        self.From = set()
        self.Neighbor = set() # for undirected edges
    def GetTo(self):
        return copy.deepcopy(self.To)
    def GetFrom(self):
        return copy.deepcopy(self.From)
    def GetNeighbor(self):
        return copy.deepcopy(self.Neighbor)
    def AddTo(self, x):
        self.To.add(x)
    def AddFrom(self, x):
        self.From.add(x)
    def AddNeighbor(self, x):
        self.Neighbor.add(x)
    def DelTo(self, x):
        self.To.remove(x)
    def DelFrom(self, x):
        self.From.remove(x)
    def DelNeighbor(self, x):
        self.Neighbor.remove(x)

class DiGraph(object):
    def __init__(self, generatedAdjmat=None):
        self.DirectedEdges = set()
        adjmat = generatedAdjmat
        self.NodeIDs = list(range(len(adjmat))) # ordered list
        self.Nodes = {i: Node() for i in self.NodeIDs}
        for i in self.NodeIDs:
            for j in self.NodeIDs:
                if adjmat[i, j]: self.add_di_edge(i, j)

    def is_adjacent(self, fromnode, tonode):
        return self.has_di_edge(fromnode, tonode) or self.has_di_edge(tonode, fromnode)

    def has_di_edge(self, fromnode, tonode):
        return (fromnode, tonode) in self.DirectedEdges

    def add_di_edge(self, fromnode, tonode):
        if not self.has_di_edge(fromnode, tonode):
            self.DirectedEdges.add((fromnode, tonode))
            self.Nodes[fromnode].AddTo(tonode)
            self.Nodes[tonode].AddFrom(fromnode)

    def del_di_edge(self, fromnode, tonode):
        if self.has_di_edge(fromnode, tonode):
            self.DirectedEdges.remove((fromnode, tonode))
            self.Nodes[fromnode].DelTo(tonode)
            self.Nodes[tonode].DelFrom(fromnode)

    def getTo(self, x):
        return self.Nodes[x].GetTo()

    def getFrom(self, x):
        return self.Nodes[x].GetFrom()

    def getAncestorsOfNode(self, x):
        ancestors = set()
        q = []
        q.append(x)
        while len(q) > 0:
            now = q.pop(0)
            for nowsparent in self.getFrom(now):
                ancestors.add(nowsparent)
                q.append(nowsparent)
        return ancestors

    def getAncestorsOfNode_no_passing(self, x, nopassingset):
        ancestors = set()
        q = []
        q.append(x)
        while len(q) > 0:
            now = q.pop(0)
            for nowsparent in self.getFrom(now):
                if nowsparent in nopassingset: continue
                ancestors.add(nowsparent)
                q.append(nowsparent)
        return ancestors

    def getAncestorsOfSet(self, Xset):
        X_ancestors = set().union(*[self.getAncestorsOfNode(x) for x in Xset])
        return X_ancestors.union(Xset)

    def getAncestorsOfSetOutsideS(self, Xset, S):
        X_ancestors_no_passing_S = set().union(*[self.getAncestorsOfNode_no_passing(x, S) for x in Xset])
        return X_ancestors_no_passing_S.union(Xset)

    def existsEffectsFromS1ToS2(self, S1, S2):
        if not S1.isdisjoint(S2): return True
        for s1 in S1:
            for s2 in S2:
                if s1 in self.getAncestorsOfNode(s2):
                    return True
        return False

    def existsEffectsFromS1ToS2WithoutPassingS3(self, S1, S2, S3):
        if not S1.isdisjoint(S2): return True
        for s1 in S1:
            for s2 in S2:
                if s1 in self.getAncestorsOfNode_no_passing(s2, S3):
                    return True
        return False

    def find_minimum_vertex_cut_size_from_AncZ_to_Y(self, AncZ, Y):
        select_S_from = [list(copi) for copi in
                chain.from_iterable(combinations(self.NodeIDs, r) for r in range(len(self.NodeIDs)))]

        for subset in select_S_from:
            Y_minus_S = set(Y) - set(subset)
            if self.existsEffectsFromS1ToS2WithoutPassingS3(set(AncZ), Y_minus_S, subset): continue
            return len(subset)