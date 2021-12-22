We provide a demo for our registration method. 

The code can be easily executed by "bash run.sh".

Users can modify "img1_path" and "img2_path" to load the pairwise images. 

SIFT and SuperPoint+SuperGlue matching strategies are supported, please specify "method" as "SIFT" or "SuperGlue" as you wish.

The reprojection error estimation is supported, which first projects the keypoints of selected inliers on the source image to the target image using the estimated affine matrix 
and then computes MSE between the projected locations and the counterparts of the corresponding keypoints on the target image. One could use this as the evaluation matric, but
it might be uneffective in some scenarios where the preserved correspondences (selected inliers) are actually dominated by false matches.


