### Deep Homography

A homography dataset can be constructed by taking a patch from an image and perturbing the corners. The 4 point correspondences between the original patch and the perturbed ones define a homography. A neural network can learn to produce the parameterized homography given the patches from the warped and unwarped image. 

This repo reproduces the work of [DeTone et al. 2016](https://arxiv.org/pdf/1606.03798.pdf). Check out my [blog post](https://ekrim.github.io/computer/vision,pytorch,homography/2018/08/07/deep-homography-estimation.html) for more analysis.

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
./make_homography_data ../unlabeled2017 <int_patch_size> <int_max_jitter> <n_samples>
```

Use a patch size of 128, a max jitter of 32, and 500,000 samples to reproduce my results.

To train the network (saves each epoch):

```
python main.py
```

To see the performance on, e.g., the 4th image:

```
python eval.py --i 3
```
![Original image, warped image, and original image warped with predicted homography](https://ekrim.github.io/assets/good_img_2.png)
