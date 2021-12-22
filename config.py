import argparse

parser = argparse.ArgumentParser(
        description='Registration demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

arg_lists = []

def add_argument_group(name):
	arg = parser.add_argument_group(name)
	arg_lists.append(arg)
	return arg

data_arg = add_argument_group("Data")
data_arg.add_argument(
	'--img1_path', type=str, default='',
	help='File path for the source image')
data_arg.add_argument(
	'--img2_path', type=str, default='',
	help='File path for the target image')
data_arg.add_argument(
	'--ransac_thr', type=float, default=10,
	help='Threshold of RANSAC')
data_arg.add_argument(
	'--num_kpt', type=int, default=8000,
	help='Number of SIFT keypoints')
data_arg.add_argument(
	'--size', type=int, default=1024,
	help='Resolution of image')
data_arg.add_argument(
	'--laplacian', action='store_true',
	help='Laplacian features')
data_arg.add_argument(
	'--method', type=str, default='SuperGlue',
	help='Feature matching method')

sift_arg = add_argument_group("SIFT")
sift_arg.add_argument(
	'--contrastThreshold', type=float, default=1e-5,
	help='Contrast threshold for SIFT')
sift_arg.add_argument(
	'--rt_thr', type=float, default=0.8,
	help='Threshold of ratio test for SIFT')

parser.add_argument(
    '--input', type=str, default='0',
    help='ID of a USB webcam, URL of an IP camera, '
         'or path to an image directory or movie file')
parser.add_argument(
    '--output_dir', type=str, default=None,
    help='Directory where to write output frames (If None, no output)')

sg_arg = add_argument_group("SuperGlue")
sg_arg.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
sg_arg.add_argument(
    '--max_keypoints', type=int, default=-1,
    help='Maximum number of keypoints detected by Superpoint'
         ' (\'-1\' keeps all keypoints)')
sg_arg.add_argument(
    '--keypoint_threshold', type=float, default=0.001,
    help='SuperPoint keypoint detector confidence threshold')
sg_arg.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
sg_arg.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
sg_arg.add_argument(
    '--match_threshold', type=float, default=-1,
    help='SuperGlue match threshold')
sg_arg.add_argument(
    '--show_keypoints', action='store_true',
    help='Show the detected keypoints')
sg_arg.add_argument(
    '--no_display', action='store_true',
    help='Do not display images to screen. Useful if running remotely')
sg_arg.add_argument(
    '--force_cpu', action='store_true',
    help='Force pytorch to run in CPU mode.')

def get_config():
	config, unparsed = parser.parse_known_args()
	return config, unparsed

def print_usage():
	parser.print_usage()


