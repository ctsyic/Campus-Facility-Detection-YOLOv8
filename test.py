import os
from ultralytics import YOLO


def run_final_test():
    # 1. 自动获取当前项目的绝对路径
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 2. 设定模型路径
    model_path = os.path.join(base_path, 'runs', 'detect', 'runs', 'train', 'campus_v1', 'weights', 'best.pt')

    # 3. 设定测试照片文件夹路径
    input_folder = os.path.join(base_path, 'my_test_images')

    # 路径安全检查
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件！路径：{model_path}")
        return
    if not os.path.exists(input_folder):
        print(f"错误：找不到照片文件夹！路径：{input_folder}")
        return

    # 4. 加载模型
    print("正在加载南阳理工校园设施识别模型...")
    model = YOLO(model_path)

    # 5. 执行推理（24张照片批量处理）
    print(f"检测到待测文件夹：{input_folder}")
    print("⏱正在进行 AI 实时侦测，请稍候...")

    results = model.predict(
        source=input_folder,
        save=True,  # 保存带框结果图
        conf=0.4,  # 置信度阈值
        project='runs/detect',
        name='final_presentation_results',
        exist_ok=True  # 覆盖同名文件夹，保持目录整洁
    )

    # 6. 遍历并打印结果（解决变量未使用的警告，并丰富录屏内容）
    print("\n" + "-" * 30)
    print("详细检测报告：")
    for result in results:
        # 获取文件名
        file_name = os.path.basename(result.path)
        # 获取检测到的目标数量
        detection_count = len(result.boxes)
        print(f"图片 [{file_name}]: 成功识别 {detection_count} 个设施")
    print("-" * 30)

    print("\n" + "=" * 50)
    print("任务圆满完成！")
    print(f"结果保存路径：{os.path.join(base_path, 'runs', 'detect', 'final_presentation_results')}")
    print("=" * 50)


if __name__ == '__main__':
    run_final_test()
