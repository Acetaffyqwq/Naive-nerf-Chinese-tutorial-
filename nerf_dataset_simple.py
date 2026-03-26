import os
import json
import torch
import numpy as np
import cv2


def get_rays(H, W, focal, c2w):
    """
    计算光线的原点和方向
    H, W: 图像的高和宽
    focal: 焦距
    c2w: 相机到世界的变换矩阵 (4x4)
    """
    # 生成网格坐标 (u, v)
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), 
        torch.linspace(0, H - 1, H), 
        indexing='ij'
    )
    i = i.t()  # 转置以匹配 (H, W)
    j = j.t()
    
    # 归一化相机坐标系下的方向向量 (x, y, z)
    # NeRF 使用右手坐标系，相机看向 -z 方向
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], 
        dim=-1
    )
    
    # 将方向向量从相机坐标系转换到世界坐标系
    # dirs: (H, W, 3) -> (H, W, 1, 3)
    # c2w[:3, :3]: (3, 3)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    
    # 光线原点即相机在世界坐标系下的位置
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d


def load_blender_data(basedir, split='train', load_imgs=True):
    """
    加载 Blender 格式的数据 (如 lego)
    """
    with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
        meta = json.load(fp)

    imgs = []
    poses = []
    
    for frame in meta['frames']:
        # 读取位姿 (c2w)
        poses.append(np.array(frame['transform_matrix']))
        if load_imgs:
            # 读取图片
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)  # 读取包括 Alpha 通道
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img / 255.0).astype(np.float32) 
            imgs.append(img)

    if load_imgs:
        imgs = np.stack(imgs)
    else:
        imgs = None
    poses = np.array(poses).astype(np.float32)
    
    # 获取图像尺寸，如果没加载图片则从元数据中推断或设默认值
    if load_imgs:
        H, W = imgs[0].shape[:2]
    else:
        # Blender 数据默认通常是 800x800
        H, W = 800, 800
        
    # 计算焦距
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    return imgs, poses, [H, W, focal]


def sample_random_pixels(imgs, poses, hwf, num_samples=1024):
    """
    从数据集中随机采样若干个像素点，并返回其光线和颜色
    """
    H, W, focal = hwf
    num_imgs = len(imgs)
    
    # 随机选择一张图片
    img_idx = np.random.randint(0, num_imgs)
    img = torch.from_numpy(imgs[img_idx])
    pose = torch.from_numpy(poses[img_idx])
    
    # 计算该张图片的所有光线
    rays_o, rays_d = get_rays(H, W, focal, pose)
    
    # 将图像和光线拉平，方便采样
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    target = img.reshape(-1, 4 if img.shape[-1] == 4 else 3)  # 处理 RGBA 或 RGB
    
    # 随机选择像素索引
    select_inds = np.random.choice(H * W, size=num_samples, replace=False)
    
    # 获取采样点的光线和颜色
    batch_rays_o = rays_o[select_inds]
    batch_rays_d = rays_d[select_inds]
    batch_target = target[select_inds]
    
    return batch_rays_o, batch_rays_d, batch_target


if __name__ == "__main__":
    # 示例运行
    data_dir = 'data/lego'
    
    print(f"Loading data from {data_dir}...")
    images, poses, hwf = load_blender_data(data_dir, 'train')
    H, W, focal = hwf
    print(f"Loaded {len(images)} images, resolution: {H}x{W}, focal: {focal}")
    
    # 采样 10 个像素点作为演示
    num_samples = 10
    rays_o, rays_d, target_rgb = sample_random_pixels(images, poses, hwf, num_samples=num_samples)
    
    print("\nSampled 10 points:")
    for i in range(num_samples):
        print(f"Point {i+1}:")
        print(f"  Ray Origin: {rays_o[i].numpy()}")
        print(f"  Ray Direction: {rays_d[i].numpy()}")
        print(f"  Pixel Color (RGBA/RGB): {target_rgb[i].numpy()}")
