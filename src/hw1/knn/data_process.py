import torch
import torchvision
import torchvision.transforms as transforms


class2name = [
    'airplane',  # 0
    'automobile',  # 1
    'bird',  # 2
    'cat',  # 3
    'deer',  # 4
    'dog',  # 5
    'frog',  # 6
    'horse',  # 7
    'ship',  # 8
    'truck'  # 9
]


# 定义数据预处理：将图像转为Tensor，并进行归一化处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行归一化
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 使用 DataLoader 来批量加载训练数据和测试数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True)


train_data_iter = iter(trainloader)
train_images, train_labels = next(train_data_iter)
X_train = train_images.cpu().numpy()
y_train = train_labels.cpu().numpy()

test_data_iter = iter(testloader)
test_images, test_labels = next(test_data_iter)
X_test = test_images.cpu().numpy()
y_test = test_labels.cpu().numpy()


# 查看数据形状
if True:
    print(f"Train images shape: {X_train.shape}")  # (50000, 3, 32, 32)
    print(f"Train labels shape: {y_train.shape}")  # (50000,)
    print(f"Test images shape: {X_test.shape}")    # (10000, 3, 32, 32)
    print(f"Test labels shape: {y_test.shape}")    # (10000,)

X_train = X_train.reshape(50000, -1)
X_test = X_test.reshape(10000, -1)

if False:
    print(f"Train images shape after flatten: {X_train.shape}")  # (50000, 3072)
    print(f"Train labels shape after flatten: {y_train.shape}")  # (50000,)
    print(f"Test images shape after flatten: {X_test.shape}")    # (10000, 3072)
    print(f"Test labels shape after flatten: {y_test.shape}")    # (10000,)
