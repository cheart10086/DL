import os
import torchvision
from torchvision import datasets, transforms

def save_mnist_by_class(dataset, save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    digit_dirs = [save_dir+'/'+str(i) for i in range(10)]
    for dir in digit_dirs:
        os.makedirs(dir, exist_ok=True)

    # 遍历数据集并保存到对应的类别目录
    for i, (image, label) in enumerate(dataset):
        digit = label
        save_path = os.path.join(digit_dirs[digit], f"mnist_image_{i}.png")
        torchvision.utils.save_image(image, save_path)

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)

# 按类别保存MNIST数据集
save_mnist_by_class(mnist_dataset, 'mnist_by_class')