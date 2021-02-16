# ImageNet
The notebooks are meant to be read to gain insights and adapted to your dataset. 
Running the notebooks require work such as preparing ImageNet data and 
checkpoints during training. To read the notebook, it is easier to read from 
colab by clicking the **Open In Colab** badge due to file size.   

## If you really want to run the colab/notebook
* The notebook is prepared with Colab + TPU. To convert it for GPUs, please  
  replace TPU strategy with GPU strategy. For details, please refer to [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy).
* Running with Colab can be easier since it provides free TPU quota.
* Download [ImageNet](http://www.image-net.org/) and load with [TensorFlow Datasets](https://www.tensorflow.org/datasets). Please refer  to [tfds imagenet2012](https://www.tensorflow.org/datasets/catalog/imagenet2012) 
  for instructions. In the notebooks, we iterate tfds for calculation and 
  visualize by reading raw data. 
* [Train ResNet-50](https://github.com/frederick0329/TrackIn/blob/master/imagenet/resnet50/trainer.py) and save the [checkpoints](https://www.tensorflow.org/guide/checkpoint) of each epoch. 

## Application of TrackIn
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frederick0329/TrackIn/blob/master/imagenet/resnet50_imagenet_self_influence.ipynb) Inspecting Training data with Self-Influence
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frederick0329/TrackIn/blob/master/imagenet/resnet50_imagenet_proponents_opponents.ipynb) Identifying Influential Training Points (Proponents and Opponents) of a Test point 
  * Consider replacing the brute-force the nearest neighbor with [ScaNN](https://github.com/google-research/google-research/tree/master/scann) to run on full ImageNet instead of 10%. 

## References

* [Advances in Neural Information Processing Systems 33 (NeurIPS 2020)](https://papers.nips.cc/paper/2020)
    - [Estimating Training Data Influence by Tracing Gradient Descent](https://papers.nips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html)  Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan
    - [TracIn — A Simple Method to Estimate Training Data Influence](https://ai.googleblog.com/2021/02/tracin-simple-method-to-estimate.html) 
      Google AI Blog
* [Cluster analysis](https://en.wikipedia.org/wiki/Cluster_analysis)
* [Kernel method](https://en.wikipedia.org/wiki/Kernel_method)
* [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* [Similarity measure](https://en.wikipedia.org/wiki/Similarity_measure)
* [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

