class Node:
    def __init__(self, opt_attr_id, opt_attr_name, depth=0, father=None):
        self.opt_attr_id = opt_attr_id
        self.depth = depth
        self.opt_attr_name = opt_attr_name
        self.child = dict()

        # used in pruning
        self.father = father

    def __call__(self, data):
        if len(self.child) == 0:
            print('error: no child')

        return self.child[data[self.opt_attr_id]](data)

    def isRoot(self):
        return self.depth == 0

    def isLeaf(self):
        return False

    def __repr__(self):
        str = f'Used Attribute: {self.opt_attr_name}\n'
        for cnt, (cdk, cdv) in enumerate(self.child.items()):
            if cnt == len(self.child) - 1:
                if type(cdv) is Leaf:
                    str += f'{self.depth * "│ "}└ {self.opt_attr_name}{cdk} -> {cdv}'
                    # str += f'\n{self.depth * "│ "}'
                else:
                    str += f'{(self.depth + 1) * "│ "}{self.opt_attr_name}{cdk} -> {cdv}'
                    # str += f'\n{self.depth * "│ "}'
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

    def isLeaf(self):
        return True

    def isRoot(self):
        return self.depth == 0
