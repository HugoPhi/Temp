import matplotlib.pyplot as plt


def display_images(images, labels, rows=1, cols=1):  # by OpenAI ChatGPT-4o
    """
    显示来自numpy数组的图像，并在每张图片下方显示对应的标签。

    :param images: 包含图像的列表或numpy数组，每个元素是一个图像。
    :param labels: 与images一一对应的标签列表。
    :param rows: 显示图像的行数。
    :param cols: 显示图像的列数。
    """
    figure, axarr = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # 调整图形大小以适应标签

    count = 0
    for row in range(rows):
        for col in range(cols):
            if count < len(images):
                if rows == 1 and cols == 1:
                    axarr.imshow(images[count])
                    axarr.set_xlabel(f'{labels[count]}')
                elif rows == 1 or cols == 1:  # 单行列情况
                    if rows == 1:
                        ax_to_use = axarr[col]
                    else:
                        ax_to_use = axarr[row]
                    ax_to_use.imshow(images[count])
                    ax_to_use.set_xlabel(f'{labels[count]}')
                else:
                    axarr[row, col].imshow(images[count])
                    axarr[row, col].set_xlabel(f'{labels[count]}')
                count += 1
            else:
                if rows > 1 and cols > 1:
                    axarr[row, col].axis('off')
                elif rows == 1 or cols == 1:
                    if rows == 1:
                        axarr[col].axis('off')
                    else:
                        axarr[row].axis('off')

    plt.suptitle("K nearest neighbors for CIFAR-10")  # 使用suptitle代替title，避免影响单个子图
    plt.tight_layout()  # 自动调整子图参数，为总标题留出空间
    plt.savefig(f'./task4/k={col}.png')
    # plt.show()
