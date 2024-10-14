import node
import numpy as np


def OptAttr(data, label, attr_dict):
    def Ent(label):
        prob = np.bincount(label) / len(label)
        return -np.sum(prob * np.log2(prob + 1e-10))

    def Gain(label, attr, attr_val):
        gain = Ent(label)
        for val in attr_val:
            label_temp = label[data[:, attr] == val]
            if len(label_temp) == 0:
                continue
            gain -= len(label_temp) / len(data) * Ent(label_temp)
        return gain

    attr = np.argmax([Gain(label, i, attr_val) for i, attr_val in enumerate(attr_dict.values())])
    attr_val = list(attr_dict.values())[attr]
    aattr_dict = {k: v for j, (k, v) in enumerate(attr_dict.items()) if j != attr}  # remove the selected attribute
    return attr, attr_val, aattr_dict


def ID3(data, label, attr_dict, pre_val=None, valid=None, valid_label=None, accuracy=None, root=None, pruning='none'):
    if (label[0] == label).all():
        return node.Leaf(np.bincount(label).argmax(), pre_val)

    if len(attr_dict) == 0:
        return node.Leaf(np.bincount(label).argmax(), pre_val)

    aflag = True
    for attr_ids in range(len(attr_dict)):
        if len(data[:, attr_ids]) == 0:
            continue
        if not (data[0, attr_ids] == data[:, attr_ids]).all():
            aflag = False
            break

    if aflag is True:
        return node.Leaf(np.bincount(label).argmax(), pre_val)

    opt_attr_id, attr_vals, attr_dict_without_opt_attr = OptAttr(data, label, attr_dict)

    tree = node.Node(opt_attr_id, pre_val, accuracy, root)
    if tree.isRoot():
        tree.root = tree

    for attr_val in attr_vals:
        data_of_same_attrval = data[data[:, opt_attr_id] == attr_val]
        label_of_same_attrval = label[data[:, opt_attr_id] == attr_val]

        if pruning == 'pre':  # TODO: pre-pruning
            pass

        if (len(label_of_same_attrval) == 0):
            tree.child[attr_val] = node.Leaf(np.bincount(label).argmax(), attr_val)
        else:
            tree.child[attr_val] = ID3(data_of_same_attrval.copy(), label_of_same_attrval.copy(), attr_dict_without_opt_attr.copy(), attr_val, valid, valid_label, accuracy, tree.root, pruning)

    if pruning == 'post' and tree.isRoot():  # TODO: post-pruning
        pass

    return tree
