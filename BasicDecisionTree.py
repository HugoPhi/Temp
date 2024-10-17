import node
import copy
import numpy as np


class DecisionTree:
    def __init__(self, data, label, attr_dict, key2id=None, depth=0, valid=None, valid_label=None, pruning='none'):
        # self attributes
        self.data = data
        self.label = label
        self.attr_dict = attr_dict
        self.key2id = key2id
        self.depth = depth
        self.valid = valid
        self.valid_label = valid_label
        self.pruning = pruning

        # tree
        self.tree = None

    def fit(self):
        self.tree = self.build(
            data=self.data,
            label=self.label,
            attr_dict=self.attr_dict,
            key2id=self.key2id,
            depth=self.depth,
            valid=self.valid,
            valid_label=self.valid_label,
            father=None,
            pruning=self.pruning)
        return self.tree

    def __call__(self, data):
        if data.ndim == 1:
            return self.tree(data)
        else:
            return np.array([self.tree(x) for x in data])

    def build(self, data, label, attr_dict, key2id=None, depth=0, valid=None, valid_label=None, father=None, pruning='none'):
        if father is None:
            key2id = {key: idx for idx, key in enumerate(attr_dict.keys())}
        if (label[0] == label).all():
            return node.Leaf(label[0], depth)

        if len(attr_dict) == 0:
            return node.Leaf(np.bincount(label).argmax(), depth)

        aflag = True
        for attr_ids in [key2id[k] for k in attr_dict.keys()]:
            if (data[0, attr_ids] != data[:, attr_ids]).any():
                aflag = False
                break

        if aflag is True:
            return node.Leaf(np.bincount(label).argmax(), depth)

        opt_attr_id, attr_vals, attr_dict_without_opt_attr = self.opt_attr(data, label, attr_dict, key2id)
        opt_attr_name = opt_attr_id
        opt_attr_id = key2id[opt_attr_id]

        tree = node.Node(
            opt_attr_id=opt_attr_id,
            opt_attr_name=opt_attr_name,
            father=father,
            depth=depth)

        tree.father = tree

        if not pruning == 'none':
            pre_accuracy = np.mean(np.bincount(label).argmax() == label)

        for attr_val in attr_vals:
            label_of_same_attrval = label[data[:, opt_attr_id] == attr_val]

            if (len(label_of_same_attrval) == 0):
                tree.child[attr_val] = node.Leaf(np.bincount(label).argmax(), depth + 1)
            else:
                tree.child[attr_val] = node.Leaf(np.bincount(label_of_same_attrval).argmax(), depth + 1)

        after_precision = 0
        if pruning == 'pre':  # Strategy: Maximize metric
            after_precision = self.metric(valid, valid_label, tree)
            if not pre_accuracy < after_precision:
                return node.Leaf(np.bincount(label).argmax(), depth)

        for attr_val in attr_vals:
            data_of_same_attrval = data[data[:, opt_attr_id] == attr_val]
            label_of_same_attrval = label[data[:, opt_attr_id] == attr_val]
            valid_of_same_attrval = valid[valid[:, opt_attr_id] == attr_val]
            valid_label_of_same_attrval = valid_label[valid[:, opt_attr_id] == attr_val]

            if (len(label_of_same_attrval) == 0):
                tree.child[attr_val] = node.Leaf(np.bincount(label).argmax(), depth + 1)
                continue

            tree.child[attr_val] = self.build(
                data=data_of_same_attrval.copy(),
                label=label_of_same_attrval.copy(),
                attr_dict=attr_dict_without_opt_attr.copy(),
                key2id=key2id,
                valid=valid_of_same_attrval.copy(),
                valid_label=valid_label_of_same_attrval.copy(),
                father=tree,
                pruning=pruning,
                depth=depth + 1)

        if pruning == 'post' and tree.isRoot():
            tree = self.post_pruning(valid, valid_label, tree)

        return tree

    def metric(self, data, label, tree):
        '''
        Pre-Pruning Strategy: Maximize metric
        '''
        res = []
        for x in data:
            res.append(tree(x))
        res = np.array(res)
        return np.mean(res == label)

    def post_pruning(self, valid, valid_label, tree_node, root=None):
        '''
        Post-Pruning Strategy: Maximize metric
        '''
        root = root
        all_children_are_leaf = True

        if tree_node.isRoot():
            root = tree_node

        for key, child in tree_node.child.items():
            if not child.isLeaf():
                tree_node.child[key] = self.post_pruning(valid, valid_label, child, root)
                all_children_are_leaf = False

        if all_children_are_leaf:
            pre_precision = self.metric(valid, valid_label, root)
            tree_copy = copy.deepcopy(tree_node)

            tree_node = node.Leaf(np.bincount(valid_label).argmax(), tree_node.depth)
            post_precision = self.metric(valid, valid_label, root)
            if pre_precision > post_precision:
                tree_node = tree_copy
            else:
                return tree_node

        return tree_node

    def opt_attr(self, data, label, attr_dict, key2id):
        '''
        Based on information gain
        '''
        def Ent(label):
            prob = np.bincount(label) / len(label)
            res = np.array([p * np.log2(p) if p != 0 else 0 for p in prob])
            return -np.sum(res)

        def Gain(label, attr, attr_val):
            gain = Ent(label)
            for val in attr_val:
                label_temp = label[data[:, attr] == val]
                if len(label_temp) == 0:
                    continue
                gain -= len(label_temp) / len(label) * Ent(label_temp)
            return gain

        attr = np.argmax([Gain(label, key2id[key], attr_val) for key, attr_val in attr_dict.items()])
        attr_val = list(attr_dict.values())[attr]
        aattr_dict = {k: v for j, (k, v) in enumerate(attr_dict.items()) if j != attr}  # remove the selected attribute

        attr = list(attr_dict.keys())[attr]  # the name of the selected attribute
        return attr, attr_val, aattr_dict
