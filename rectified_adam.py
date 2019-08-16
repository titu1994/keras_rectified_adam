from keras import backend as K
from keras.optimizers import Optimizer


# Ported from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
class RectifiedAdam(Optimizer):
    """RectifiedAdam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        final_lr: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: Weight decay weight.
        amsbound: boolean. Whether to apply the AMSBound variant of this
            algorithm.

    # References
        - [On the Variance of the Adaptive Learning Rate and Beyond]
          (https://arxiv.org/abs/1908.03265)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.0, **kwargs):
        super(RectifiedAdam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

        self.weight_decay = float(weight_decay)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            beta2_t = self.beta_2 ** t
            N_sma_max = 2 / (1 - self.beta_2) - 1
            N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)

            # apply weight decay
            if self.weight_decay != 0.:
                p_wd = p - self.weight_decay * lr * p
            else:
                p_wd = None

            if p_wd is None:
                p_ = p
            else:
                p_ = p_wd

            def gt_path():
                step_size = lr * K.sqrt(
                    (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max /
                    (N_sma_max - 2)) / (1 - self.beta_1 ** t)

                denom = K.sqrt(v_t) + self.epsilon
                p_t = p_ - step_size * (m_t / denom)

                return p_t

            def lt_path():
                step_size = lr / (1 - self.beta_1 ** t)
                p_t = p_ - step_size * m_t

                return p_t

            p_t = K.switch(N_sma > 5, gt_path, lt_path)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'weight_decay': self.weight_decay}
        base_config = super(RectifiedAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Ported from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
# class EMARectifiedAdam(Optimizer):
#     """EMARectifiedAdam optimizer.
#
#     Default parameters follow those provided in the original paper.
#
#     # Arguments
#         lr: float >= 0. Learning rate.
#         final_lr: float >= 0. Final learning rate.
#         beta_1: float, 0 < beta < 1. Generally close to 1.
#         beta_2: float, 0 < beta < 1. Generally close to 1.
#         gamma: float >= 0. Convergence speed of the bound function.
#         epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
#         decay: float >= 0. Learning rate decay over each update.
#         weight_decay: Weight decay weight.
#         amsbound: boolean. Whether to apply the AMSBound variant of this
#             algorithm.
#
#     # References
#         - [On the Variance of the Adaptive Learning Rate and Beyond]
#           (https://arxiv.org/abs/1908.03265)
#         - [Adam - A Method for Stochastic Optimization]
#           (https://arxiv.org/abs/1412.6980v8)
#         - [On the Convergence of Adam and Beyond]
#           (https://openreview.net/forum?id=ryQu7f-RZ)
#     """
#
#     def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
#                  epsilon=None, decay=0., weight_decay=0.0,
#                  buffer=None, **kwargs):
#         super(EMARectifiedAdam, self).__init__(**kwargs)
#
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.lr = K.variable(lr, name='lr')
#             self.beta_1 = K.variable(beta_1, name='beta_1')
#             self.beta_2 = K.variable(beta_2, name='beta_2')
#             self.decay = K.variable(decay, name='decay')
#
#         if epsilon is None:
#             epsilon = K.epsilon()
#         self.epsilon = epsilon
#         self.initial_decay = decay
#
#         self.weight_decay = float(weight_decay)
#
#         if buffer is None:
#             self.buffer = [[None, None, None]
#                            for _ in range(10)]
#         else:
#             self.buffer = buffer
#
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]
#
#         lr = self.lr
#         if self.initial_decay > 0:
#             lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
#                                                       K.dtype(self.decay))))
#
#         t = K.cast(self.iterations, K.floatx()) + 1
#         t_int = K.cast(self.iterations, 'int64') + 1
#         buffered = self.buffer[(t_int % 10)]
#
#         ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
#         vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
#         self.weights = [self.iterations] + ms + vs
#
#         for p, g, m, v in zip(params, grads, ms, vs):
#             m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
#             v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
#
#             if t_int == buffered[0]:
#
#                 N_sma, step_size = buffered[1], buffered[2]
#
#             else:
#                 buffered[0] = t_int
#
#                 beta2_t = self.beta_2 ** t
#                 N_sma_max = 2 / (1 - self.beta_2) - 1
#                 N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)
#                 buffered[1] = N_sma
#
#                 def gt_step_path():
#                     step_size = lr * K.sqrt(
#                         (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max /
#                         (N_sma_max - 2)) / (1 - self.beta_1 ** t)
#
#                     return step_size
#
#                 def lt_step_path():
#                     step_size = lr / (1 - self.beta_1 ** t)
#
#                     return step_size
#
#                 step_size = K.switch(N_sma > 5, gt_step_path, lt_step_path)
#                 buffered[2] = step_size
#
#             # apply weight decay
#             if self.weight_decay != 0.:
#                 p_wd = p - self.weight_decay * lr * p
#             else:
#                 p_wd = None
#
#             if p_wd is None:
#                 p_ = p
#             else:
#                 p_ = p_wd
#
#             def gt_sma_path():
#                 denom = K.sqrt(v_t) + self.epsilon
#                 p_t = p_ - step_size * (m_t / denom)
#
#                 return p_t
#
#             def lt_sma_path():
#                 p_t = p_ - step_size * m_t
#
#                 return p_t
#
#             p_t = K.switch(N_sma > 4, gt_sma_path, lt_sma_path)
#
#             self.updates.append(K.update(m, m_t))
#             self.updates.append(K.update(v, v_t))
#             new_p = p_t
#
#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)
#
#             self.updates.append(K.update(p, new_p))
#         return self.updates
#
#     def get_config(self):
#         buffer = [[]]
#
#         config = {'lr': float(K.get_value(self.lr)),
#                   'beta_1': float(K.get_value(self.beta_1)),
#                   'beta_2': float(K.get_value(self.beta_2)),
#                   'decay': float(K.get_value(self.decay)),
#                   'epsilon': self.epsilon,
#                   'weight_decay': self.weight_decay,
#                   'buffer': self.buffer}
#         base_config = super(EMARectifiedAdam, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
