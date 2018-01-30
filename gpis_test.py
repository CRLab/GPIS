import pcl
import tempfile
import argparse
import subprocess
import curvox.cloud_conversions
import curvox.cloud_to_mesh_conversions
import plyfile

import numpy as np

import matlab.engine


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_depth_pcd_filepath', type=str,
                    help='input pcd filepath for caputured depth')
parser.add_argument('--input_tactile_pcd_filepath', type=str,
                    help='input pcd filepath for caputured tactile')
parser.add_argument('-o', "--output_obj_filepath", type=str,
                    help='output mesh file')

parser.add_argument('-d', '--depth_downsample_factor', default=20,  #200 pts # 300pts #400pts?
                    help='Factor by which to downsample the observed depth cloud')

parser.add_argument('--resolution', default=40.0, #40, 64, 100
					type=float,
                    help='Voxel resolution to use during sampling of GPIS')

args = parser.parse_args()


depth_cloud = pcl.PointCloud()
depth_cloud.from_file(args.input_depth_pcd_filepath)
tactile_cloud = pcl.PointCloud()
tactile_cloud.from_file(args.input_tactile_pcd_filepath)

vt_fhandle, vt_fpath = tempfile.mkstemp(suffix=".pcd")
points, normals = curvox.cloud_conversions.calculate_normals_from_depth_and_tactile(depth_cloud, tactile_cloud, args.depth_downsample_factor)
curvox.cloud_conversions.write_pcd_with_normals(points, normals, vt_fpath)

eng = matlab.engine.start_matlab()
eng.gpis(vt_fpath, args.output_obj_filepath, args.resolution, 0.005, 0.005, 0.001, nargout=0) #0.005, 0.005, 0.001,   0.001, 0.001, 0.0005

curvox.cloud_to_mesh_conversions.convert_obj_to_ply(args.output_obj_filepath, args.output_obj_filepath.replace(".obj", ".ply"))
