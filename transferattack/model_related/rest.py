import os
import imageio
from PIL import Image
import numpy as np
from tqdm import tqdm

input_dir = "track_figs"
output_filename = input_dir + '.mp4'
output_filepath = os.path.join(os.getcwd(), output_filename)

# 找第一张图片确定统一尺寸
first_img = None
for file_name in sorted(os.listdir(input_dir)):
    if file_name.endswith('.png'):
        first_img = Image.open(os.path.join(input_dir, file_name)).convert('RGB')
        break

if first_img is None:
    raise ValueError("未找到 PNG 文件")

target_size = first_img.size  # (width, height)

images = []
for file_name in tqdm(sorted(os.listdir(input_dir)), desc="Processing images"):
    if file_name.endswith('.png'):
        try:
            img = Image.open(os.path.join(input_dir, file_name)).convert('RGB')
            if img.size != target_size:
                img = img.resize(target_size)
            frame_array = np.array(img)
            if frame_array.ndim != 3 or frame_array.shape[2] != 3:
                print(f"⚠️ 图片 {file_name} 维度异常 {frame_array.shape}，跳过")
                continue
            images.append(frame_array)
        except Exception as e:
            print(f"⚠️ 读取图片 {file_name} 失败，错误：{e}")

fps = 30  # 每秒30帧

with imageio.get_writer(output_filepath, fps=fps, codec='libx264') as writer:
    for frame in tqdm(images, desc="Writing video"):
        writer.append_data(frame)

print(f'✅ 视频已保存: {output_filepath}')
