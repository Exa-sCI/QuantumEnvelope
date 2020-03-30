#!/usr/bin/env python3

# Explanation:
# Computing the reuse distance of a trace involves figuring out, for each
# key k at time t, the number of keys accessed since the last access to k.
# An efficient way to compute this is to use a balanced binary tree, indexed by
# the last time of each access. This way, it is enough to find a node in the
# tree and compute the number of nodes with a bigger key.
# The key can then be removed and reinserted with its new access time.

# Splay Trees:
# Splay tree are a type of balanced trees that have the interesting property of
# pushing the last key accessed at the root of the tree. This makes computing
# the number of nodes with a bigger key easy: it's the weight of the right
# subtree.

# We thus implement a splay tree that maintains internal metadata about the
# weight is each subtree, and use it for reuse distance computations.

from typing import Tuple, NewType, Optional, Any

AccessTime = NewType('AccessTime', int)
Coordinates = Tuple[int, int, int, int]


class Node:
    '''Node of a Splay Tree: contains a key, value, weight,
    and links to left, right, parent'''

    def __init__(self, key: AccessTime, value: Any, weight: int,
                 left: Optional['Node'], right: Optional['Node'],
                 parent: Optional['Node']):
        self.key = key
        self.value = value
        self.weight = weight
        self.left = left
        self.right = right
        self.parent = parent

    def __str__(self):
        return f"{self.key}:{self.value}:{self.weight}:" + repr(self.parent)


class SplayTree:
    '''Binary Self-Balancing Tree that pushes the last accessed value at the
    root of the tree through "splaying".'''

    def __init__(self):
        self.root = None

    def __rotate_right(self, p: Node):
        #            P                 Q
        #           / \       --->    / \
        #          Q   C             A   P
        #         / \                   / \
        #        A   B                 B   C
        q = p.left
        assert q is not None

        # move B to left of P
        p.left = q.right
        if q.right is not None:
            q.right.parent = p
        q.right = p

        # move Q in place of P
        if p.parent is not None:
            if p == p.parent.left:
                p.parent.left = q
            else:
                p.parent.right = q
        q.parent = p.parent
        p.parent = q

        # recompute the weights
        p.weight = 1
        if p.left is not None:
            p.weight += p.left.weight
        if p.right is not None:
            p.weight += p.right.weight

        q.weight = 1 + p.weight
        if q.left is not None:
            q.weight += q.left.weight

        # if P was the root, update it
        if self.root == p:
            self.root = q

    def __rotate_left(self, p: Node):
        #     P                    Q
        #    / \                  / \
        #   A   Q    ---->       P   C
        #      / \              / \
        #     B   C            A   B
        q = p.right
        assert q is not None

        # move B to right of P
        p.right = q.left
        if p.right is not None:
            p.right.parent = p
        q.left = p

        # move Q in place of P
        if p.parent is not None:
            if p == p.parent.left:
                p.parent.left = q
            else:
                p.parent.right = q
        q.parent = p.parent
        p.parent = q

        # recompute the weights
        p.weight = 1
        if p.left is not None:
            p.weight += p.left.weight
        if p.right is not None:
            p.weight += p.right.weight

        q.weight = 1 + p.weight
        if q.right is not None:
            q.weight += q.right.weight

        # if P was the root, update it
        if self.root == p:
            self.root = q

    def splay(self, n: Node):
        '''See wikipedia for explanation'''

        if self.root == n:
            return

        while n.parent is not None:
            if n.parent.parent is None:
                # zig
                if n == n.parent.left:
                    self.__rotate_right(self.root)
                else:
                    self.__rotate_left(self.root)
            else:
                p = n.parent
                g = p.parent
                assert p is not None and g is not None
                # zig-zig
                if n == p.left and p == g.left:
                    self.__rotate_right(g)
                    self.__rotate_right(p)
                elif n == p.right and p == g.right:
                    self.__rotate_left(g)
                    self.__rotate_left(p)
                # zig-zag
                elif n == p.right and p == g.left:
                    self.__rotate_left(p)
                    self.__rotate_right(g)
                elif n == p.left and p == g.right:
                    self.__rotate_right(p)
                    self.__rotate_left(g)

    def __insert_recursive(self, root: Optional[Node], parent: Optional[Node],
                           key: AccessTime, value: Any) -> Node:
        if root is None:
            n = Node(key, value, 1, None, None, parent)
            assert parent is not None
            if key < parent.key:
                parent.left = n
            else:
                parent.right = n
            return n
        else:
            root.weight += 1
            if key < root.key:
                return self.__insert_recursive(root.left, root, key, value)
            else:
                return self.__insert_recursive(root.right, root, key, value)

    def insert(self, key: AccessTime, value: Any):
        '''Insert a new node in the tree. Splay after insertion.'''

        # we single out the empty case because we don't need to splay
        if self.root is None:
            self.root = Node(key, value, 1, None, None, None)
            return

        n = self.__insert_recursive(self.root, None, key, value)
        self.splay(n)

    def find(self, key: AccessTime) -> Optional[Node]:
        '''Find a node in the tree. Splay if found.'''

        cur = self.root
        while cur is not None:
            if key == cur.key:
                break
            elif key < cur.key:
                cur = cur.left
            else:
                cur = cur.right
        if cur is not None:
            self.splay(cur)
        return cur

    def __change_link_in_parent(self, n: Node, k):
        p = n.parent
        if p is None:
            return
        if n == p.left:
            p.left = k
        else:
            p.right = k

    def remove_node(self, n: Node):
        if n.left is None and n.right is None:
            if self.root == n:
                self.root = None
            self.__change_link_in_parent(n, None)
        elif n.left is not None and n.right is None:
            # simple case, only a left child
            if self.root == n:
                self.root = n.left
                n.left.parent = None
            self.__change_link_in_parent(n, n.left)
            n.left.parent = n.parent
        elif n.left is None and n.right is not None:
            # only right child
            if self.root == n:
                self.root = n.right
                n.right.parent = None
            self.__change_link_in_parent(n, n.right)
            n.right.parent = n.parent
        else:
            # complex case: replace the node by its in-order successor, and
            # apply the above method.
            cur = n.right
            assert cur is not None
            while cur.left is not None:
                cur.weight -= 1
                cur = cur.left

            # exchange
            n.key = cur.key
            n.value = cur.value
            n.weight -= 1
            # and delete cur
            self.remove_node(cur)

    def remove(self, key: AccessTime):
        '''Delete a node. Splay its in-order predecessor.'''
        # search and splay
        n = self.find(key)
        if n is None:
            return

        self.remove_node(n)

    def __str_aux__(self, n: Optional[Node], l: int) -> str:
        if n is None:
            return ''
        else:
            return f"{n.key}:{n.value}\n" \
                    + l * " " + "L:" + self.__str_aux__(n.left, l+1) + "\n" \
                    + l * " " + "R:" + self.__str_aux__(n.right, l+1)

    def __str__(self):
        return self.__str_aux__(self.root, 1)


class ReuseDistance:

    def __init__(self):
        self.record = SplayTree()
        self.last_seen = {}
        self.time = 0

    def add_record(self, record):
        self.time += 1
        distance = -1
        key = self.last_seen.get(record, -1)
        if key != -1:
            node = self.record.find(key)
            assert node is not None
            assert node.value == record
            assert self.record.root == node
            if node.right is None:
                distance = 0
            else:
                distance = node.right.weight
            self.record.remove(key)
        self.record.insert(self.time, record)
        self.last_seen[record] = self.time
        assert self.record.root.weight == len(self.last_seen.keys())
        return distance


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=argparse.FileType('r'),
                        help='Trace file')

    args = parser.parse_args()

    r = ReuseDistance()
    for line in args.trace:
        coords = line.strip()
        d = r.add_record(coords)
        print(coords, d)
