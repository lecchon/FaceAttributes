

import keras
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.metrics import mean_absolute_error, categorical_accuracy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np

import sys
import os
import json
import argparse
import random

from model import build
from reader import DataLoader
from losses import smooth_l1,focal_loss
from export import *
#from metrics import category_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train mult-attributes classifier')
    parser.add_argument('--dataset', type=str, 
        default='./data', help='The root of dataset to train')
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt',
        help='the directory to store the params')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='the base learning rate of model')
    parser.add_argument('--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight_decay', type=float, default=0.0002,
        help='the weight_decay of net')
    parser.add_argument('--train_backbone', type=bool, default=True,
        help='whether train backbone or not')
    parser.add_argument('--pretrain', type=str, default=None,
        help='init net from pretrained model default is None')
    parser.add_argument('--num_epoches', type=int, default=30,
        help='max iters to train network, default is 30')
    parser.add_argument('--trainable_branch', type=str, default="all",
        help='which branch should be trained, comma splited."all" for the three branches.')
    parser.add_argument('--feature_aggregate', type=bool, default=False,
        help='whether do feature aggregation.')
    parser.add_argument('--base_model', type=str, default="mobilenetv2",
        choices=["MobileNetV2","DenseNet121","Xception","VGG16","VGG19"], help='select a backbone model')
    parser.add_argument('--iter_size', type=int, default=100,
        help='iter size equal to the batch size, default 100')
    parser.add_argument('--loss_weights', type=str, default="1.6,0.8,1.2",
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--log', type=str, default='log',
        help='the file to store log, default is log.txt')
    parser.add_argument('--batch_size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--val_size', type=int, default=100,
        help='size of validation set, default 100')
    parser.add_argument('--crop_face', type=bool, default=True,
        help='whether crop face.')
    parser.add_argument('--labels', type=str, default=None,
        help='label .json file')
    parser.add_argument('--image_size', type=int, default=224,
        help='input image size.')
    return parser.parse_args()



def train():
    ## parse args
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    with open("config.json") as fi:
        config=json.load(fi)
    ## define model
    features = config.get(args.base_model) if args.feature_aggregate else None
    model = build(input_shape=(args.image_size,args.image_size,3), 
                  base_model=args.base_model,
                  feature_layers=features, 
                  train_backbone=args.train_backbone,
                  weight_decay=args.weight_decay)
    ## select branch(s) to train
    if args.trainable_branch != "all":
        ## freeze untrainable branch
        branches = set(["age","gender","race"])
        trainable_branches = set(args.trainable_branch.split(","))
        untrainable_branches = branches.symmetric_difference(trainable_branches)
        for i in range(len(model.layers)):
            for b in untrainable_branches:
                if model.layers[i].output.name.startswith("branch_"):
                    if model.layers[i].output.name.startswith("branch_"+b):
                        model.layers[i].trainable = False
        
    ## compile
    loss_weights = [float(i) for i in args.loss_weights.split(",")]
    model.compile(optimizer=Adam(args.lr), 
                  loss={"age":smooth_l1,
                        "gender":categorical_crossentropy,
                        "race":categorical_crossentropy},
                  loss_weights={"age":loss_weights[0],
                                "gender":loss_weights[1],
                                "race":loss_weights[2]},
                  metrics={"age":mean_absolute_error,
                           "gender":categorical_accuracy,
                           "race":categorical_accuracy})
    ## load pretrain
    if args.pretrain is not None:
        if not os.path.exists(args.pretrain):
            print("[！] `%s` doesn't exists!" % args.pretrain)
            sys.exit(1)
        else:
            print("[...] loading weights from %s" % args.pretrain)
            model.load_weights(args.pretrain, skip_mismatch=True)
    ## read data
    if args.labels is not None:
        with open(args.labels) as fi:
            label = json.load(fi)
    else:
        label = None
    file_list = [os.path.join(args.dataset,f) for f in os.listdir(args.dataset) if f.endswith("jpg")]
    random.shuffle(file_list)
    train_data = DataLoader(file_list=file_list[:-args.val_size], image_size=args.image_size, labels=label, 
                            mode="train", argument=True, crop_face=args.crop_face)
    val_data = DataLoader(file_list=file_list[-args.val_size:], image_size=args.image_size, labels=label, 
                            mode="train", argument=False, crop_face=args.crop_face)
    ## start training
    print("[...] start training")
    logging = TensorBoard(log_dir=args.log,update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(args.checkpoint_dir,"model{epoch:03d}_val_loss{val_loss:.3f}.h5"),
                                 monitor="val_loss", save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)
    model.fit_generator(train_data.flow(args.batch_size), 
                        steps_per_epoch=args.iter_size, 
                        epochs=args.num_epoches, 
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping], 
                        validation_data=val_data.flow(args.batch_size), 
                        validation_steps=int(args.val_size//args.batch_size),
                        initial_epoch=0,
                        use_multiprocessing=False,
                        workers=0)
    ## save checkpoint
    model.save_weights(os.path.join(args.checkpoint_dir,"model_final.h5"))
    ## export pb file
    export_pb(sess, [i.op.name for i in model.inputs], [i.op.name for i in model.outputs], "./model", "model_final")
    ## export coreml
    export_coreml(model, model.input.op.name, [i.op.name for i in model.outputs], "./model", "model_final")
    ## export tflite
    export_tflite(sess, model.inputs, model.outputs, "./model", "model_final")
    
    
if __name__ == "__main__":
    train()
    