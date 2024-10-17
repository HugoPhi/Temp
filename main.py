import numpy as np
# from sklearn import tree
# import decisionTree as dt
import BasicDecisionTree as dt

# watermelon
attr_dict = {
    '色泽': ['青绿', '乌黑', '浅白'],
    '根蒂': ['蜷缩', '稍蜷', '硬挺'],
    '敲声': ['浊响', '沉闷', '清脆'],
    '纹理': ['清晰', '稍糊', '模糊'],
    '脐部': ['凹陷', '稍凹', '平坦'],
    '触感': ['硬滑', '软粘']
}

data = np.array([
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘'],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑'],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑']
])

# Labels for "好瓜" (0 = 否, 1 = 是)
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# shuffle
# shuffle_ix = np.random.permutation(len(data))
# data = data[shuffle_ix]
# labels = labels[shuffle_ix]

train_ix = np.array([1, 2, 3, 6, 7, 10, 14, 15, 16, 17]) - 1
valid_ix = np.array([4, 5, 8, 9, 11, 12, 13]) - 1
train_data = np.array([data[i] for i in train_ix])
train_labels = np.array([labels[i] for i in train_ix])
valid_data = np.array([data[i] for i in valid_ix])
valid_labels = np.array([labels[i] for i in valid_ix])

# v1.0
# print('mine')
# tree = dt.ID3(train_data, train_labels, attr_dict, valid=valid_data, valid_label=valid_labels, pruning='post')
# res = []
# for x in valid_data:
#     res.append(tree(x))
# print(np.array(res))
# print(valid_labels)
# print('mine acc: ', np.mean(res == valid_labels))
# print()
# print('tree is: ')
# print(tree)


# v2.0
for way in ['none', 'pre', 'post']:
    print(f'mine: {way}')
    tree = dt.DecisionTree(train_data, train_labels, attr_dict, valid=valid_data, valid_label=valid_labels, pruning=way)
    tree.fit()

    res = tree(valid_data)
    print(res)
    print(valid_labels)
    print('mine acc: ', np.mean(res == valid_labels))
    print('tree is: ')
    print(tree.tree)
    print()
