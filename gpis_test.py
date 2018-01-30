import pcl
import tempfile

# Define Variables
pcd_filename = './data/inputs/Bunny_ascii.pcd'
obj_filename = './data/outputs/Bunny.obj'

# Convert PCD to PCD with normals
pcd = pcl.PointCloud()
pcd.from_file(pcd_filename)
points = pcd.to_array()
normals = pcd.calc_normals(ksearch=10, search_radius=0)

new_pcd_filename = "tmp.pcd" #tempfile.mktemp(suffix=".pcd")
with open(new_pcd_filename, 'w') as new_pcd_file:
	new_pcd_file.write(
		"# .PCD v0.7 - Point Cloud Data file format\n"
		"VERSION 0.7\n"
		"FIELDS x y z normal_x normal_y normal_z\n"
		"SIZE 4 4 4 4 4 4\n"
		"TYPE F F F F F F\n"
		"COUNT 1 1 1 1 1 1\n"
		"WIDTH {0}\n"
		"HEIGHT 1\n"
		"POINTS {0}\n"
		"DATA ascii\n"
		.format(points.shape[0])
	)

	for point, normal in zip(points, normals):
		new_pcd_file.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], normal[0], normal[1], normal[2]))


exit()

import matlab.engine
eng = matlab.engine.start_matlab()
eng.gpis(new_pcd_filename, obj_filename, 40, nargout=0)

