# Image-Captioning using VGG for feature extraction

Using Flickr8k dataset 1GB. for each photo 5 descriptions are available. 

Used [Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/) backend for the code. **VGG** is used for extracting the features.

No Beam search is yet implemented.

You can download the weights [here](https://github.com/yashk2810/Image-Captioning/raw/master/weights/time_inceptionV3_2.8876_loss.h5)

# Examples
!["epoch1"](https://bitbucket.org/teodor_cotet/imagecaptioning/raw/758986caaa054ec7437f9bd10f2c9c0e2f5071ec/results/photos/epoch1.PNG)
!["epoch7"](https://bitbucket.org/teodor_cotet/imagecaptioning/raw/758986caaa054ec7437f9bd10f2c9c0e2f5071ec/results/photos/epoch7.PNG)
!["epoch12"](https://bitbucket.org/teodor_cotet/imagecaptioning/raw/758986caaa054ec7437f9bd10f2c9c0e2f5071ec/results/photos/epoch12.PNG)

# Dependencies

* Keras 1.2.2
* Tensorflow 0.12.1
* numpy
* matplotlib

# References

[1] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. [Show and Tell: A Neural Image Caption Generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)

[2] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). [VGG](https://arxiv.org/pdf/1409.1556.pdf%20http://arxiv.org/abs/1409.1556.pdf)