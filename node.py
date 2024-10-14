class Node:
    def __init__(self, opt_attr_id, opt_attr_name, depth=0, accuracy=None, root=None):
        self.opt_attr_id = opt_attr_id
        self.depth = depth
        self.opt_attr_name = opt_attr_name
        self.child = dict()

        # used in pruning
        self.accuracy = accuracy
        self.root = root

    def __call__(self, data):
        if len(self.child) == 0:
            print('error: no child')

        return self.child[data[self.opt_attr_id]](data)

    def isRoot(self):
        return self.root is None

    def __repr__(self):
        str = f'Used Attribute: {self.opt_attr_name}\n'
        for cnt, (cdk, cdv) in enumerate(self.child.items()):
            if cnt == len(self.child) - 1:
                str += f'{self.depth * "│ "}└ {self.opt_attr_name}{cdk} -> {cdv}'
            else:
                str += f'{(self.depth + 1) * "│ "}{self.opt_attr_name}{cdk} -> {cdv}\n'
        return str


class Leaf:
    def __init__(self, label, depth=0):
        self.label = label
        self.depth = depth

    def __call__(self, data):
        return self.label

    def __repr__(self):
        return f'Class: {self.label}'
