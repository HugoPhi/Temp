import numpy as np
# from sklearn import tree
import decisionTree as dt


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
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑']
])

# Labels for "好瓜" (0 = 否, 1 = 是)
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

shuffle_ix = np.random.permutation(len(data))
data = data[shuffle_ix]
labels = labels[shuffle_ix]


# mine
print('mine')
tree = dt.ID3(data, labels, attr_dict)
res = []
for x in data:
    res.append(tree(x))
print(np.array(res))
print(labels)
print('tree is: ')
print(tree)


# sklearn
# print('sklearn')
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(data, labels)
# res = []
# for x in data:
#     res.append(clf.predict([x]))
# print(res)


attr_dict = {}
