import numpy as np

class IntervalRBTree:
    class Node:
        def __init__(self, item=None, int_max=-np.Inf, is_red=False, left=None, right=None, parent=None):
            self.item = item
            self.int_max_source = int_max
            self.is_red = is_red
            self.left = left
            self.right = right
            self.parent = parent

        @property
        def int_max(self):
            if self.item is not None and isinstance(self.int_max_source, type(self.item)):
                return self.int_max_source[1]
            else:
                return self.int_max_source

        @int_max.setter
        def int_max(self, value):
            self.int_max_source = value

        @int_max.deleter
        def int_max(self):
            del self.int_max_source

    def __init__(self, items=None):
        self._items = items
        self.size = 0
        self._root = None
        self._nil = self.Node()

        if items is not None:
            for item in items:
                self.insert(item)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)

    def __str__(self):
        x = self._root
        color = 'R' if x.is_red else 'B'
        draw = '(' + color + ':' + str(x.item) + ':' + str(x.int_max) + ')\n'
        if x.right != self._nil:
            draw = self._traverse_tree(x.right, draw, '  ', False)
        if x.left != self._nil:
            draw = self._traverse_tree(x.left, draw, '  ', True)
        return draw

    def _traverse_tree(self, x, draw, pad, is_left):
        color = 'R' if x.is_red else 'B'
        draw += pad
        if is_left or x.parent.left == self._nil:
            pad += '    '
            draw += ' └──'
        else:
            pad += ' |   '
            draw += ' ├──'
        draw += '(' + color + ':' + str(x.item) + ':' + str(x.int_max) + ')\n'

        if x.right != self._nil:
            draw = self._traverse_tree(x.right, draw, pad, False)
        if x.left != self._nil:
            draw = self._traverse_tree(x.left, draw, pad, True)
        return draw

    def minimum(self):
        x = self._minimum(self._root)
        return x.item[0], x.item

    def maximum(self):
        x = self._maximum(self._root)
        return x.item[0], x.item

    def _minimum(self, x):
        while x.left != self._nil:
            x = x.left
        return x

    def _maximum(self, x):
        while x.right != self._nil:
            x = x.right
        return x

    def insert(self, item):

        if self.size == 0:
            self._root = self.Node(item, int_max=item, left=self._nil, right=self._nil, parent=self._nil)
            self.size += 1
            return

        z = self.Node(item, int_max=item, is_red=True, left=self._nil, right=self._nil)
        y = self._nil
        x = self._root
        # go down the tree to where the new node needs to be placed.
        while x != self._nil:
            y = x
            x.int_max = [x.int_max_source if x.int_max > z.item[1] else z.item][0]
            if z.item < x.item:
                x = x.left
            else:
                x = x.right
        z.parent = y
        if y == self._nil:
            self._root = z
        elif z.item < y.item:
            y.left = z
        else:
            y.right = z
        self._color_fixup(z)
        self.size += 1

    def _color_fixup(self, z):
        while z.parent.is_red:
            if z.parent == z.parent.parent.left:
                child1, child2 = 'left', 'right'
            else:
                child1, child2 = 'right', 'left'

            y = getattr(z.parent.parent, child2)
            if y.is_red:
                z.parent.is_red = False
                y.is_red = False
                z.parent.parent.is_red = True
                z = z.parent.parent
            else:
                if z == getattr(z.parent, child2):
                    z = z.parent
                    getattr(self, '_'+child1+'_rotate')(z)
                z.parent.is_red = False
                z.parent.parent.is_red = True
                getattr(self, '_'+child2+'_rotate')(z.parent.parent)
        self._root.is_red = False

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        x.int_max = [x.item if x.item[1] > max(x.left.int_max, x.right.int_max) else
                     [x.left.int_max_source if x.left.int_max > x.right.int_max else x.right.int_max_source][0]][0]
        if y.left != self._nil:
            y.left.parent = x
        y.parent = x.parent
        if y.parent == self._nil:
            self._root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        y.int_max = [y.item if y.item[1] > max(y.left.int_max, y.right.int_max) else
                     [y.left.int_max_source if y.left.int_max > y.right.int_max else y.right.int_max_source][0]][0]
        x.parent = y

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        x.int_max = [x.item if x.item[1] > max(x.left.int_max, x.right.int_max) else
                     [x.left.int_max_source if x.left.int_max > x.right.int_max else x.right.int_max_source][0]][0]
        if y.right != self._nil:
            y.right.parent = x
        y.parent = x.parent
        if y.parent == self._nil:
            self._root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.right = x
        y.int_max = [y.item if y.item[1] > max(y.left.int_max, y.right.int_max) else
                     [y.left.int_max_source if y.left.int_max > y.right.int_max else y.right.int_max_source][0]][0]
        x.parent = y

    def search(self, i):
        x = self._root
        while x != self._nil and not self._overlap(x.item, i):
            if x.left != self._nil and x.left.int_max >= i[0]:
                x = x.left
            else:
                x = x.right
        return x.item

    @staticmethod
    def _overlap(i1, i2):
        if i1[0] == i1[1]:
            return (i2[0] <= i1[0]) and (i1[0] < i2[1])
        if i2[0] == i2[1]:
            return (i1[0] <= i2[0]) and (i2[0] < i1[1])

        return ((i1[1] < i2[1]) and (i1[1] > i2[0])) or ((i2[1] < i1[1]) and (i2[1] > i1[0])) or \
               ((i1[0] <= i2[0]) and (i2[1] <= i1[1])) or ((i2[0] <= i1[0]) and (i1[1] <= i2[1]))

    def search_exactly(self, item):
        x = self._root
        while x != self._nil:
            if item == x.item:
                return x
            elif item < x.item:
                x = x.left
            else:
                x = x.right
        return None

    def delete(self, item):
        z = self.search_exactly(item)
        if z is None:
            return
        y = z
        y_is_red = y.is_red
        if z.left == self._nil:
            x = z.right
            self._transplant(z, z.right)  # check if you can change with x (clearer)
        elif z.right == self._nil:
            x = z.left
            self._transplant(z, z.left)  # check if you can change with x (clearer)
        else:
            y = self._minimum(z.right)
            y_is_red = y.is_red
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self._transplant(y, y.right) # this line does not cause problems
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.is_red = z.is_red
            y.int_max = [y.item if y.item[1] > max(y.left.int_max, y.right.int_max) else
                         [y.left.int_max_source if y.left.int_max > y.right.int_max else y.right.int_max_source][0]][0]

        # fix the red-black colors
        if not y_is_red:
            self._delete_fixup(x)
        self.size -= 1

        # fix the int_max upstream
        while z.parent != self._nil:
            z = z.parent
            z.int_max = [z.item if z.item[1] > max(z.left.int_max, z.right.int_max) else
                         [z.left.int_max_source if z.left.int_max > z.right.int_max else z.right.int_max_source][0]][0]

    def _transplant(self, u, v):
        if u.parent == self._nil:
            self._root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_fixup(self, x):
        while x != self._root and not x.is_red:
            if x == x.parent.left:
                child1, child2 = 'left', 'right'
            else:
                child1, child2 = 'right', 'left'

            w = getattr(x.parent, child2)
            if w.is_red:
                w.is_red = False
                x.parent.is_red = True
                getattr(self, '_'+child1+'_rotate')(x.parent)
                w = getattr(x.parent, child2)
            if not getattr(w, child1).is_red and not getattr(w, child2).is_red:
                w.is_red = True
                x = x.parent
            else:
                if not getattr(w, child2).is_red:
                    getattr(w, child1).is_red = False
                    w.is_red = True
                    getattr(self, '_'+child2+'_rotate')(w)
                    w = getattr(x.parent, child2)
                w.is_red = x.parent.is_red
                x.parent.is_red = False
                getattr(w, child2).is_red = False
                getattr(self, '_'+child1+'_rotate')(x.parent)
                x = self._root
        x.is_red = False


class Interval:
    def __init__(self, low=None, high=None):
        self.low = low
        self.high= high

    def __getitem__(self, it):
        if it:
            return self.high
        else:
            return self.low

    def __lt__(self, other):
        return (self[0] - other[0] < -1e-12) or (abs(self[1] - other[0]) < 1e-12)

    def __str__(self):
        return '[' + str(self.low) + ',' + str(self.high) + ']'


if __name__ == '__main__':
    #intervals = [(6, 10), (17, 19), (19, 20), (26, 26), (25, 30), (16, 21), (0, 3), (8, 9), (15, 33), (5, 8)]
    intervals = [(6, 10), (17, 19), (19, 20), (26, 26), (25, 30), (16, 21), (0, 3), (8, 9), (15, 23), (5, 8)]
    #intervals = [(1, 3), (3, 4), (9.2, 10.5), (4, 6.6), (6.6, 7), (7, 9), (9, 9.2)]
    #items = [(5,6), (5, 6.7), (5, 9), (5, 5.2), (5, 5), (5, 6.9)]
    #a, b = (15, 23), (22, 25)
    #extended = [b] + intervals + items + [a]
    # think I left here with fixing the interval red black tree.
    items = [Interval(*interval) for interval in intervals]
    hold = items[-2]
    tree = IntervalRBTree(items=items)
    print(tree)
    tree.delete(hold)
    print(tree)
