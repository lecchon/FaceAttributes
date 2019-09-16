import keras
from keras.layers import *
from keras.applications import MobileNetV2,DenseNet121,VGG16,VGG19,Xception
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

model_factory = {
    "MobileNetV2": MobileNetV2,
    "DenseNet121": DenseNet121,
    "vgg16": VGG16,
    "VGG19": VGG19,
    "Xception": Xception,
}

def ConvBNReLU(x, filters, ksize, strides, padding="same", weight_decay=0.001):
    x = SeparableConv2D(filters=filters, kernel_size=ksize, strides=strides,
                        padding=padding, use_bias=False, 
                        depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    return x

def attention(x, weight_decay=0.001):
    filters = x.shape.as_list()[-1]
    q = SeparableConv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False,
                        depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
    q = BatchNormalization()(q)
    q = Activation("sigmoid")(q)
    x = Multiply()([q,x])
    return x

    
def build(input_shape, base_model="mobilenetv2", feature_layers=None, train_backbone=False, weight_decay=0.0001):
    inputs = Input(input_shape)
    base = model_factory[base_model]
    backbone = base(input_tensor=inputs, weights="imagenet", include_top=False)
    base_out = backbone.output
    backbone.trainable = train_backbone
    if feature_layers is not None:
        features = []
        for fea in feature_layers:
            features.append(backbone.get_layer(fea).output)
        with tf.variable_scope("master"):
            fea = ConvBNReLU(features[0], features[1].shape.as_list()[-1], 3, 2, weight_decay=weight_decay)
            fea = Concatenate(axis=-1)([fea, features[1]])
            fea = ConvBNReLU(fea, features[2].shape.as_list()[-1], 3, 2, weight_decay=weight_decay)
            ## modify
            fea = ConvBNReLU(fea, base_out.shape.as_list()[-1], 1, 1, weight_decay=weight_decay)
            fea = Concatenate(axis=-1)([fea, base_out])
            fea = ConvBNReLU(fea, 128, 1, 1, weight_decay=weight_decay)
    else:
        fea = base_out
        
    with tf.variable_scope("branch_age"):
        #age_score = ConvBNReLU(fea, 256, 3, 2, weight_decay=weight_decay)
        age_score = fea
        age_score = attention(age_score, weight_decay)
        age_score = SeparableConv2D(32, 1, 1, depthwise_regularizer=l2(weight_decay), 
                            pointwise_regularizer=l2(weight_decay))(age_score)
        age_score = GlobalAvgPool2D()(age_score)
        age_score = Dense(1)(age_score)
        age_score = ReLU(10,name="age")(age_score)
        
    with tf.variable_scope("branch_gender"):
        #gender_score = ConvBNReLU(fea, 256, 3, 2, weight_decay=weight_decay)
        gender_score = fea
        gender_score = attention(gender_score, weight_decay)
        gender_score = SeparableConv2D(64, 1, 1, depthwise_regularizer=l2(weight_decay), 
                            pointwise_regularizer=l2(weight_decay))(gender_score)
        gender_score = GlobalAvgPool2D()(gender_score)
        gender_score = Dense(2)(gender_score)
        gender_score = Softmax(name="gender")(gender_score)

    with tf.variable_scope("branch_race"):
        #race_score = ConvBNReLU(fea, 256, 3, 2, weight_decay=weight_decay)
        race_score = fea
        race_score = attention(race_score, weight_decay)
        race_score = SeparableConv2D(64, 1, 1, depthwise_regularizer=l2(weight_decay), 
                            pointwise_regularizer=l2(weight_decay))(race_score)
        race_score = GlobalAvgPool2D()(race_score)
        race_score = Dense(5)(race_score)
        race_score = Softmax(name="race")(race_score)
    
    return Model(input=inputs, output=[age_score,gender_score,race_score])