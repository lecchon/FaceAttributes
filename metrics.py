from keras import backend as K

def category_accuracy(y_true, y_pred):
    pred_class = K.cast(K.squeeze(K.argmax(y_pred,-1), axis=-1), K.floatx())
    if K.ndim(y_true) == 1:
        y_true = y_true
    elif y_true.shape.as_list()[1] == 1:
        y_true = K.squeeze(y_true, axis=1)
    else:
        y_true = K.argmax(y_true, 1)
    true_class = K.cast(y_true, K.floatx())
    return K.cast(K.equal(pred_class, true_class), K.floatx())