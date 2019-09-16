from keras import backend as K
import tensorflow as tf

## define smooth L1
def smooth_l1(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_part = K.cast(K.less(diff, 1), tf.float32)
    greater_part = 1.0 - less_part
    return less_part*K.pow(diff,2) + greater_part*(K.abs(0.5-diff))

## define focal loss
def focal_loss(y_true, y_pred, gamma=0.5, epsilon=1e-5):
    #y_true = K.one_hot(K.cast(y_true, tf.int32), y_pred.shape.as_list()[-1])
    pos_term = -y_true * K.pow(1-y_pred,gamma) * K.log(K.clip(y_pred, epsilon, 1.0))
    neg_term = -(1-y_true) * K.pow(y_pred,gamma) * K.log(K.clip(1-y_pred, epsilon, 1.0))
    return pos_term+neg_term