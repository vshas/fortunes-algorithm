import numpy as np


class Heap:
    def __init__(self, items=None, max_items=1000):
        self._items = items
        self._keys = np.ones(max_items) * -np.Inf
        self._indexes = np.array(range(max_items))

        if items is not None:
            self._keys[:len(items)] = np.array([item.key for item in items])
            self._heapify()

    def __len__(self):
        if self._items is None:
            return 0
        else:
            return len(self._items)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)

    def __str__(self):
        index = 0
        draw = '(' + str(self._keys[index]) + ')\n'
        if 2 * index + 1 < len(self):
            draw = self._traverse_tree(draw, '  ', 2 * index + 1, 0, False)
        if 2 * index + 2 < len(self):
            draw = self._traverse_tree(draw, '  ', 2 * index + 2, 0, True)
        return draw

    def insert(self, item):
        self._keys[len(self)] = item.key
        self._items.append(item)

        possible_shift, ix = True, len(self) - 1
        while possible_shift:
            ix, possible_shift = self._shift_up(ix)

    def _shift_down(self, ix):
        if self._keys[2*ix + 1] > self._keys[2*ix + 2]:
            return 2*ix + 1, self._swap(2*ix + 1, ix)
        else:
            return 2*ix + 2, self._swap(2*ix + 2, ix)

    def _shift_up(self, ix):
        return int((ix - 1)/2), self._swap(ix, int((ix-1)/2))

    def _swap(self, i1, i2):
        if self._keys[i1] > self._keys[i2]:
            self._keys[i1], self._keys[i2] = self._keys[i2], self._keys[i1]
            self._indexes[i1], self._indexes[i2] = self._indexes[i2], self._indexes[i1]
            return True
        else:
            return False

    def find_max(self):
        return self._keys[0], self._items[self._indexes[0]]

    def delete_max(self):
        max_index = self._indexes[0]
        self._keys[0], self._keys[len(self)-1] = self._keys[len(self)-1], self._keys[0]
        self._indexes[0], self._indexes[len(self)-1] = self._indexes[len(self)-1], self._indexes[0]
        self._indexes[:len(self)][self._indexes[:len(self)] > max_index] -= 1
        self._indexes[len(self)-1] = len(self) - 1

        del self._items[max_index]
        self._keys[len(self)] = -np.Inf

        possible_shift, ix = True, 0
        while possible_shift:
            ix, possible_shift = self._shift_down(ix)

    def extract_max(self):
        max_key, max_item = self.find_max()
        self.delete_max()
        return max_key, max_item

    def _heapify(self):
        for i in reversed(range(int(len(self) / 2))):
            possible_shift = True
            ix = i
            while possible_shift:
                ix, possible_shift = self._shift_down(ix)

    def _traverse_tree(self, draw, pad, index, level, is_right):
        draw += pad
        left = True if 2*index + 1 < len(self) else False
        right = True if 2*index + 2 < len(self) else False
        if is_right or index + 1 >= len(self):
            pad += '     '
            draw += ' └──'
        else:
            pad += ' |   '
            draw += ' ├──'
        draw += '(' + str(self._keys[index]) + ') \n'
        if left:
            draw = self._traverse_tree(draw, pad, 2*index+1, level+1, False)
        if right:
            draw = self._traverse_tree(draw, pad, 2*index+2, level+1, True)
        return draw

class Event:
    def __init__(self, key, point):
        self.key = key
        self.point = point

    def __str__(self):
        return str(self.point)


def log_queue(queue, iz):
    if iz:
        print(' --------------------- ')
        print(f'keys: {queue._keys[:len(queue)]}')
        print(f'indexes: {queue._indexes[:len(queue)]}')
        print(f'items: {[str(item) for item in queue._items]}')


if __name__ == '__main__':
    points = [(2.5, 9), (5, 8.5), (8, 7), (6, 5.75), (3.5, 4), (1, 2.5), (7, 2), (9, 1.5), (3, 0.75)]
    queue = Heap([Event(key=point[1], point=point) for point in points])

    iz = 1
    log_queue(queue, iz)
    sweep_line, event = queue.extract_max()
    print(event.point)
    log_queue(queue, iz)
    sweep_line, event = queue.extract_max()
    print(event.point)
    log_queue(queue, iz)
    sweep_line, event = queue.extract_max()
    print(event.point)
    log_queue(queue, iz)
    point = (2, -13.2045)
    point2=(3, -13.2045)
    queue.insert(Event(point[1], point))
    log_queue(queue, iz)
    queue.insert(Event(point2[1], point2))
    log_queue(queue, iz)
    sweep_line, event = queue.extract_max()
    print(event.point)
    point = (4.111, 3.51658717)
    point2 = (6.334027, 5.715896)
    queue.insert(Event(point[1], point))

    log_queue(queue, iz)

    queue.insert(Event(point2[1], point2))

    log_queue(queue, iz)

    sweep_line, event = queue.extract_max()
    print(event.point)

    log_queue(queue, iz)

    print(queue)


    sweep_line, event = queue.extract_max()
    print(event.point)

