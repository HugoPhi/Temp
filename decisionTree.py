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
            gain -= len(label_temp) / len(label) * Ent(label_temp)
        return gain

    attr = np.argmax([Gain(label, i, attr_val) for i, attr_val in enumerate(attr_dict.values())])
    attr_val = list(attr_dict.values())[attr]
    aattr_dict = {k: v for j, (k, v) in enumerate(attr_dict.items()) if j != attr}  # remove the selected attribute

    attr = list(attr_dict.keys())[attr]  # the name of the selected attribute
    return attr, attr_val, aattr_dict


def ID3(data, label, attr_dict, key2id=None, depth=0, valid=None, valid_label=None, accuracy=None, root=None, pruning='none'):
    if root is None:
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

    opt_attr_id, attr_vals, attr_dict_without_opt_attr = OptAttr(data, label, attr_dict)
    opt_attr_name = opt_attr_id
    opt_attr_id = key2id[opt_attr_id]

    tree = node.Node(
        opt_attr_id=opt_attr_id,
        opt_attr_name=opt_attr_name,
        accuracy=accuracy,
        root=root,
        depth=depth)
    if tree.isRoot():
        tree.root = tree

    for attr_val in attr_vals:
        data_of_same_attrval = data[data[:, opt_attr_id] == attr_val]
        label_of_same_attrval = label[data[:, opt_attr_id] == attr_val]

        if (len(label_of_same_attrval) == 0):
            tree.child[attr_val] = node.Leaf(np.bincount(label).argmax(), depth + 1)
            continue

        if pruning == 'pre':  # TODO: pre-pruning
            pass
        else:
            tree.child[attr_val] = ID3(
                data=data_of_same_attrval.copy(),
                label=label_of_same_attrval.copy(), attr_dict=attr_dict_without_opt_attr.copy(),
                key2id=key2id,
                valid=valid,
                valid_label=valid_label,
                accuracy=accuracy,
                root=tree.root,
                pruning=pruning,
                depth=depth + 1)
    if pruning == 'post' and tree.isRoot():  # TODO: post-pruning
        pass

    return tree
