class Node:
    def __init__(self, opt_attr_id, pre_val=None, accuracy=None, root=None):
        self.opt_attr_id = opt_attr_id
        self.pre_val = pre_val
        self.child = dict()

        # used in pruning
        self.accuracy = accuracy
        self.root = root

    def __call__(self, data):
        if len(self.child):
            print('error: no child')

        for x in self.child.values():
            if data[self.opt_attr_id] == x.pre_val:
                return x(data)

    def isRoot(self):
        return self.pre_val is None


class Leaf:
    def __init__(self, label, pre_val=None):
        self.label = label
        self.pre_val = pre_val

    def __call__(self, data):
        return self.label
