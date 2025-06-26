import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from utils.options import parser
from tqdm import tqdm


def main():
    args = parser.parse_args()
    root = '/data9102/workspace/mwt/dataset/night/trafficframe'
    test_imgs = [json.loads(line) for line in open(root + '/test_pic.json')]

    # 设置目标目录为 '/data9102/workspace/mwt/dataset/night/ty'
    data_dir = '/data9102/workspace/mwt/dataset/night/ty'

    # 遍历每个图像路径
    for img_name in tqdm(test_imgs, desc="Prepare Images"):
        img_dir = img_name[0:2]

        # 创建目标目录
        temp_dir = os.path.join(data_dir, img_dir)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])

        # 确定图像路径
        img_path = os.path.join(root, img_name)  # 修正路径

        # 加载并预处理图像（如果图像文件存在）
        if os.path.exists(img_path):
            image_data = cv2.imread(img_path)  # 读取彩色图像
            if image_data is not None:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            else:
                print(f"Error: Failed to load image '{img_name}' from {img_path}")
                continue  # 跳过当前循环，处理下一个图像
        else:
            print(f"Warning: Image '{img_name}' not found at {img_path}")
            continue  # 跳过当前循环，处理下一个图像

        # 获取注视点标签数据
        lab_img, fix_x, fix_y = getLabel(root, vid_index, frame_index)

        # 将注视点添加到图像上（红色圆点）
        for x, y in zip(fix_x, fix_y):
            if 0 <= x < 720 and 0 <= y < 1280:
                # 使用 OpenCV 绘制较大的圆圈
                cv2.circle(image_data, (y, x), 10, (255, 0, 0), -1)

        # 创建一个绘图并显示图像
        plt.figure(figsize=(12, 6))
        plt.imshow(image_data)
        plt.axis('off')  # 去掉坐标轴

        # 保存图像（使用原始图像名称命名）
        output_path = os.path.join(temp_dir, f"{frame_index}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 去掉空白区域
        plt.close()  # 关闭当前的图像窗口


def getLabel(root, vid_index, frame_index):
    # 构建 .mat 文件的路径
    fixdatafile = os.path.join('/data/workspace/zcm/dataset/DrFixD-rainy/', 'fixdata', f'fixdata{vid_index}.mat')

    # 加载 .mat 文件
    if os.path.exists(fixdatafile):
        data = sio.loadmat(fixdatafile)

        # 提取固定点的坐标
        try:
            fix_x = data['fixdata'][frame_index - 1][0][:, 3]
            fix_y = data['fixdata'][frame_index - 1][0][:, 2]
        except KeyError:
            print(f"Error: Key 'fixdata' not found in {fixdatafile}")
            return None, None, None
        except IndexError:
            print(f"Error: Invalid frame index {frame_index} in {fixdatafile}")
            return None, None, None

        # 将坐标转换为整数
        fix_x = fix_x.astype('int')
        fix_y = fix_y.astype('int')

        # 创建一个空白的掩码（标签图像），大小与原始图像相同
        mask = np.zeros((720, 1280), dtype='float32')

        # 在掩码图像中标记固定点的位置
        for i in range(len(fix_x)):
            if 0 <= fix_x[i] < 720 and 0 <= fix_y[i] < 1280:
                mask[fix_x[i], fix_y[i]] = 1  # 将对应位置标记为 1（表示注视点）

        return mask, fix_x, fix_y
    else:
        print(f"Error: .mat file '{fixdatafile}' not found.")
        return None, None, None


if __name__ == '__main__':
    main()
