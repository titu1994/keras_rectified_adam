# Rectified Adam for Keras

Keras port of [Rectified Adam](https://github.com/LiyuanLucasLiu/RAdam), from the paper [On the Variance of the Adaptive Learning Rate and Beyond.](https://arxiv.org/abs/1908.03265)

# Rectified ADAM

<img src="https://github.com/titu1994/keras_rectified_adam/blob/master/images/rectified_adam.png?raw=true" height=100% width=100%>

One of the many contributions of this paper is the idea that Adam with Warmup tends to perform better than Adam without warmup. However, when Adam is used without warmup, during the initial iterations the gradients have large variance. This large variance causes overshoots of minima, and thereby leads to poor optima. 

Warmup on the other hand is the idea of training with a very low learning rate for the first few epochs to offset this large variance. However, the degree of warmup - how long and what learning rate should be used require extensive hyper parameter search, which is usually costly. 

Therefore Rectified ADAM proposes a dynamic variance reduction algorithm.

## Usage

Add the `rectified_adam.py` script to your project, and import it. Can be a dropin replacement for `Adam` Optimizer. 

Note, currently only the basic Rectified Adam is supported, not the EMA buffered variant as Keras cannot index the current timestep 
when in graph mode. This will probably be fixed in Tensorflow 2.0 Eager Execution mode.

```python
from rectified_adam import RectifiedAdam

optm = RectifiedAdam(lr=1e-3)
```


# Requirements
- Keras 2.2.4+ & Tensorflow 1.12+ (Only supports TF backend for now).
- Numpy
