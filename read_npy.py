import numpy as np
import open3d as o3d
# data = np.load('/home/hci/workspace/EscherNet/metrics/NeRF_idx/chair/train_N1M20_random.npy')

pcd = o3d.io.read_point_cloud('./demo/nerf_synthetic/chair/points3d.ply')
# data = np.load('./demo/nerf_synthetic/chair/points3d.ply')
pcd.scale(0.001, center=pcd.get_center())
data = np.array(pcd.points)

o3d.visualization.draw_geometries([pcd])

print(pcd.get_center())