# Deep Learning-based Correspondencens for Ophthalmic image registration

## Introduction
We register ophthalmic images by using a pixel-level correspondence-based registration strategy. We employ deep learning networks to locate pixel-level keypoints and match the repeatable keypoints over the pairwise images. The registration is achieved by estimating a geometric transformation from the generated correspondences and then deforming the source image based on this transformation.

## Dependencies
* python 3 >= 3.5
* pyTorch >= 1.1
* matplotlib
* numpy
* opencv-python===3.3.0.10
* imutils

One can easily install all these dependencies by executing
```
$ pip install -f requirements.txt
```

## Contents
There are three main top-level scripts in this repo:

1. `registration.py` : read pairwise images from specific files and register them by using SIFT or SuperPoint+SuperGlue.
2. `SuperGlue`: implementations of SuperPoint and SuperGlue, where the pretrained models are included. The codes are partially borrowed from the offical implementations.
3. `matching_utils.py`: include some utils about SIFT detector and descriptor, brute force matching, mutual nearest test, ratio test, image pre-processing, and visualization.

## Matching Demo Script

This demo runs SuperPoint + SuperGlue feature matching on two ophthalmic images. The results of initial matching, selected inliers, and image warping are visualized.

```
$ bash run.sh
```

One can modify the file paths of the source image and target image to register other images. `--method` should be specified as `SIFT` or `SuperGlue` to generate the corresponding results.
