import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os


def test_single_image(model_path, image_path, transform, device):
    """
    加载模型并测试单张TIF图像。

    参数:
    model_path (str): 已保存的模型权重文件路径 (.pth)。
    image_path (str): 要测试的单个TIF图像的路径。
    transform (callable): 应用于图像的torchvision变换。
    device (torch.device): 'cuda' 或 'cpu'。

    返回:
    float: 模型预测的熵值。
    """
    # 1. 初始化与训练时相同的模型架构
    model = models.resnet50(pretrained=False)  # 确保与训练时设置一致
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=1)

    # 2. 加载保存的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 3. 设置为评估模式
    model.eval()

    # 4. 打开并预处理图像
    try:
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)  # 转换为float32
        image /= 65535.0  # 归一化到 [0, 1]

        # 转换为PIL Image以应用transform
        image_pil = Image.fromarray(image)

        # 应用transformations
        image_tensor = transform(image_pil).unsqueeze(0)  # 增加batch维度 (1, 1, H, W)

    except FileNotFoundError:
        print(f"Error: The file was not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return None

    image_tensor = image_tensor.to(device)

    # 5. 执行预测
    with torch.no_grad():  # 在此模式下，不计算梯度
        normalized_output = model(image_tensor)

    # 6. 反归一化得到最终结果
    # 注意：这里的min/max值必须与训练时TifDataset类中使用的值完全相同
    label_min = 5
    label_max = 18
    predicted_value = normalized_output.item() * (label_max - label_min) + label_min

    return predicted_value


# --- 使用示例 ---

if __name__ == '__main__':
    # 定义模型和图像路径
    BEST_MODEL_PATH = r'G:\haowenjie\shang\weights\resnet50_best_model.pth'
    # 请将下面的路径替换为您想要测试的一张TIF图像的实际路径
    # 例如: r'G:\郝文杰\熵\slice-3dprint-15ep\image_001.tif'
    IMAGE_TO_TEST = r'PATH_TO_YOUR_SINGLE_IMAGE.tif'

    # 确保使用与训练时完全相同的变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 检查模型文件是否存在
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Model file not found at {BEST_MODEL_PATH}. Please check the path.")
    # 检查图像文件是否存在
    elif not os.path.exists(IMAGE_TO_TEST):
        print(f"Image file not found at {IMAGE_TO_TEST}. Please provide a valid path.")
    else:
        # 调用函数进行测试
        predicted_entropy = test_single_image(
            model_path=BEST_MODEL_PATH,
            image_path=IMAGE_TO_TEST,
            transform=test_transform,
            device=device
        )

        if predicted_entropy is not None:
            print(f"The predicted entropy for the image is: {predicted_entropy:.4f}")