# Rectified Adam for Keras

Keras port of [Rectified Adam](https://github.com/LiyuanLucasLiu/RAdam), from the paper [On the Variance of the Adaptive Learning Rate and Beyond.](https://arxiv.org/abs/1908.03265)

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
