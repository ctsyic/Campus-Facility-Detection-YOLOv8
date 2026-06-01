from ultralytics import YOLO
import torch

def train_main():
    # 1. 环境检测：确保调用你的 RTX 4070
    if torch.cuda.is_available():
        device = 0
        print(f"检测到显卡: {torch.cuda.get_device_name(0)}，即将开启高速训练！")
    else:
        device = 'cpu'
        print("未检测到显卡，将使用 CPU 训练（速度较慢）。")

    # 2. 加载预训练模型
    # 我们选择 yolov8n (nano)，它体积小、速度极快，且非常适合在教师端机器上运行
    model = YOLO('yolov8n.pt')

    # 3. 开始训练
    # 这里的参数是专门为你目前的 4 类别、小样本量优化的
    model.train(
        data='campus.yaml',      # 指定你的配置文件
        epochs=150,              # 迭代轮次，150轮足以让模型学会四类特征
        imgsz=640,               # 训练图像尺寸，640是通用标准，兼顾精度与速度
        batch=8,                 # 每一批处理的图像数量，4070可以设更高，但小样本建议8-16
        workers=0,               # 如果在 Windows 上报错，请保持为 0
        device=device,           # 指定显卡设备
        name='campus_v1',        # 训练结果保存的文件夹名
        project='runs/train',    # 结果存放的总目录
        optimizer='SGD',         # 优化器
        amp=True                 # 开启混合精度训练，加速显卡运算并节省显存
    )

if __name__ == '__main__':
    train_main()
