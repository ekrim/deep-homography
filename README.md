### Deep Homography

A homography dataset can be constructed by taking a patch from an image and perturbing the corners. The 4 point correspondences between the original patch and the perturbed ones define a homography. A neural network can learn to produce the parameterized homography given the patches from the warped and unwarped image.

#### Making the data

Download the unlabeled [MS-COCO](http://cocodataset.org/#home) dataset and unzip such that the ~330k jpeg images are in `data/unlabeled2017`.

Install the data generation code that uses OpenCV:

```
cd data
mkdir build
cmake ..
make
```

Then run the data generation program:

```
cd build
./make_homography_data ../unlabeled2017 <int_patch_size> <int_max_jitter>
```

Use a patch size of 128 and a max jitter of 30 to reproduce my results.

