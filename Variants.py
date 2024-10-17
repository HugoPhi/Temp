import BasicDecisionTree as bdt


class ID3(bdt.DecisionTree):
    def __init__(self, data, label, attr_dict, key2id=None, depth=0, valid=None, valid_label=None, pruning='none'):
        super().__init__(data, label, attr_dict, key2id, depth, valid, valid_label, pruning)
