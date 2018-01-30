## Description
# This Script loads a 3D point cloud in pcd format with its normals. 
# Then performs a 3D Reconstruction using GPIS. 

## Add libraries
import pcl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import GPy as gp
import sys
import time
import itertools
import mcubes
import plyfile

# Note to add all depndencies folders or libraries
def scatter3D(points_list, colors):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for points, color in zip(points_list, colors):
		ax.scatter(points[:,0], points[:,1], points[:,2], c=color, marker='o')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

def _generate_ply_data(points, faces):
    """

    :param points:
    :param faces:
    :return:
    """
    vertices = [(point[0], point[1], point[2]) for point in points]
    faces = [(point,) for point in faces]

    vertices_np = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces_np = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = plyfile.PlyElement.describe(vertices_np, 'vertex')
    face_element = plyfile.PlyElement.describe(faces_np, 'face')

    return plyfile.PlyData([vertex_element, face_element], text=True)


# Place matlab in the working directory

## Load 3D Object
filename_obj = './data/inputs/Bunny_ascii.pcd' # Ascii
outputfile_obj = './data/outputs/Bunny.ply'

D_frompcd = pcl.PointCloud()
D_frompcd.from_file(filename_obj)
# Format: x y z nx ny nz radius

## Prepare Data (Computing Constraints) % Assuming normals are correct ...
# Parameters can be computed automatically as a function of the size of the object(e.g. 10% of the size defined by a bounding box, see (Wendland, 2002, Surface Reconstructions from unorganized point clouds). 

d = 0.2
d_pos = 0.2
d_neg = 0.2 # 0.2
npar = 0.03 # 0.03

# Grid resolution
res = 100 # 150 # grid resolution # 50

## Computing inside and outside constraints based on normals
normals = D_frompcd.calc_normals(0.1, 20)
points = D_frompcd.to_array()

points_out = points + d_neg * normals
points_in = points - d_pos * normals

print("Expected ((166, 3), (166, 3), (166, 3))")
print(points.shape, points_in.shape, points_out.shape)

## Prepare f(x) as signace distance function
# fone=ones(1,size(points_in',1))*1;
# fminus=-1*ones(1,size(points_out',1))*1;
fone   = np.ones((points_in.shape[0], 1)) * d_pos
fminus = -1 * np.ones((points_out.shape[0], 1)) * d_neg
X      = points
fzero  = np.zeros((X.shape[0], 1))

print("Expected ((166, 1), (166, 1), (166, 3), (166, 1))")
print(fone.shape, fminus.shape, X.shape, fzero.shape)

## Visualize Object (Cube)
# Notice that the scale of the Sphere goes from -20 to 20
#scatter3D([points, points_out, points_in], ['r', 'g', 'b'])

# Training data
X = np.concatenate([X, points_in, points_out])
y = np.concatenate([fzero, fone, fminus])
print("Expected ((498, 3), (498, 1))")
print(X.shape, y.shape)

# Evaluation limits
minx = np.min(X, axis=0) - 0.6 # We extend the boundaries of the object a bit to evaluate a little bit further 
maxx = np.max(X, axis=0) + 0.6 # the 0.6 value can be adjusted dependeing the size of the bounding box, and if for example you are interested in regions outside the boundaries of the object modelled by the sensors.

print("Expected ((3,), (3,))")
print(minx.shape, maxx.shape)

# Filling the query vector
scale_x_min = minx[0]
scale_x_max = maxx[0]
scale_y_min = minx[1]
scale_y_max = maxx[1]
scale_z_min = minx[2]
scale_z_max = maxx[2]

xstar = np.zeros((res**3, 3))

for j in range(res):
	for i in range(res):
		d = j * res**2 # Delta
		axis_min = d + res * i
		axis_max = res * (i + 1) + d

		xstar[axis_min:axis_max, 0] = np.linspace(scale_x_min, scale_x_max, num=res) # in X
		xstar[axis_min:axis_max, 1] = scale_y_min + i * ((scale_y_max - scale_y_min) / res) # in X
		xstar[axis_min:axis_max, 2] = scale_z_min + ((j + 1) * ((scale_z_max - scale_z_min) / res))

tsize = res
xeva = np.reshape(xstar[:, 0], (tsize, tsize, tsize))
yeva = np.reshape(xstar[:, 1], (tsize, tsize, tsize))
zeva = np.reshape(xstar[:, 2], (tsize, tsize, tsize))

print("Expected ((100, 100, 100), (100, 100, 100), (100, 100, 100))")
print(xeva.shape, yeva.shape, zeva.shape)

# GP Setup
kern = gp.kern.RBF(input_dim=3, variance=0.01)

print("Initializing Regression model")
start = time.time()
m = gp.models.GPRegression(X, y, kern, noise_var=0.005)
print("Took {}s".format(time.time() - start))

print("Optimizing")
start = time.time()
m.optimize()
print("Took {}s".format(time.time() - start))

# Query GP
print("Predicting")
start = time.time()
prediction, _ = m.predict(xstar)
print("Took {}s".format(time.time() - start))

output_grid = np.zeros((res, res, res))
for counter, (i, j, k) in enumerate(itertools.product(range(res), range(res), range(res))):
    output_grid[i][j][k] = prediction[counter]

vertices, faces = mcubes.marching_cubes(output_grid[:, :, :], 0.5)

# Generate ply data_generate_ply_data
ply_data = _generate_ply_data(vertices, faces)
ply_data.write(open(outputfile_obj, 'wb'))

import IPython
IPython.embed()