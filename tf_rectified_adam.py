import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


class RectifiedAdam(OptimizerV2):

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 weight_decay=0.0,
                 name='RectifiedAdam', **kwargs):
        super(RectifiedAdam, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.weight_decay = weight_decay

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        t = tf.cast(self.iterations + 1, var_dtype)

        m_t = (beta_1_t * m) + (1. - beta_1_t) * grad
        v_t = (beta_2_t * v) + (1. - beta_2_t) * tf.square(grad)

        beta2_t = beta_2_t ** t
        N_sma_max = 2 / (1 - beta_2_t) - 1
        N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)

        # apply weight decay
        if self.weight_decay != 0.:
            p_wd = var - self.weight_decay * lr_t * var
        else:
            p_wd = None

        if p_wd is None:
            p_ = var
        else:
            p_ = p_wd

        def gt_path():
            step_size = lr_t * tf.sqrt(
                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max /
                (N_sma_max - 2)) / (1 - beta_1_t ** t)

            denom = tf.sqrt(v_t) + epsilon_t
            p_t = p_ - step_size * (m_t / denom)

            return p_t

        def lt_path():
            step_size = lr_t / (1 - beta_1_t ** t)
            p_t = p_ - step_size * m_t

            return p_t

        p_t = tf.cond(N_sma > 5, gt_path, lt_path)

        m_t = tf.compat.v1.assign(m, m_t)
        v_t = tf.compat.v1.assign(v, v_t)

        with tf.control_dependencies([m_t, v_t]):
            param_update = tf.compat.v1.assign(var, p_t)
            return tf.group(*[param_update, m_t, v_t])

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse data is not supported yet")

    def get_config(self):
        config = super(RectifiedAdam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
        })
        return config
