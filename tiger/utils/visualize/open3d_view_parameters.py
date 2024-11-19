import os
import numpy as np
import struct
import open3d
import time

from open3d import visualization


def save_view_point(pcd, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=2085, height=1278)
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=2085, height=1278)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    scene = 'office2'
    path = f'results/{scene}/{scene}_mesh.ply'

    pcd = open3d.io.read_point_cloud(path)  # 传入自己当前的pcd文件
    # save_view_point(pcd, "BV_1442.json")  # 保存好得json文件位置
    load_view_point(pcd, "BV_1442.json")  # 加载修改时较后的pcd文件
