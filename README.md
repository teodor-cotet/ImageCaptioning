# Image-Captioning using VGG for feature extraction

Using Flickr8k dataset 1GB. for each photo 5 descriptions are available. 

Used <a href="https://keras.io/">Keras</a> with <a href="https://www.tensorflow.org/">Tensorflow</a> backend for the code. **VGG** is used for extracting the features.

No Beam search is yet implemented.

You can download the weights <a href='https://github.com/yashk2810/Image-Captioning/raw/master/weights/time_inceptionV3_2.8876_loss.h5'>here</a>.

# Examples
Epoch 1 (on validation): 
!["epoch1"](https://bitbucket.org/teodor_cotet/imagecaptioning/raw/36670e03ded9fc9b0d5586c08d48faf74b46ef0c/results/photos/epoch1.PNG)
Epoch 7 (on validation):
!["epoch7"](https://bitbucket.org/teodor_cotet/imagecaptioning/raw/36670e03ded9fc9b0d5586c08d48faf74b46ef0c/results/photos/epoch7.PNG)
Epoch 12 (on validation):
!["epoch12"](https://bitbucket.org/teodor_cotet/imagecaptioning/raw/36670e03ded9fc9b0d5586c08d48faf74b46ef0c/results/photos/epoch12.PNG)

# Dependencies

* Keras 1.2.2
* Tensorflow 0.12.1
* numpy
* matplotlib

# References

[1] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf">Show and Tell: A Neural Image Caption Generator</a>

[2] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). <a href="https://arxiv.org/pdf/1409.1556.pdf%20http://arxiv.org/abs/1409.1556.pdf">VGG</a> 