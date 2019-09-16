from model import build
import argparse
import os
import json
from PIL import Image
import numpy as np
import dlib
from keras import backend as K
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Inference mult-attributes classifier')
    parser.add_argument('--model', type=str, 
        default=None, help='complete trained .h5 file')
    parser.add_argument('--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--feature_aggregate', type=bool, default=True,
        help='whether do feature aggregation.')
    parser.add_argument('--base_model', type=str, default="mobilenetv2",
        choices=["MobileNetV2","DenseNet121","Xception","VGG16","VGG19"], help='select a backbone model')
    parser.add_argument('--crop_face', type=bool, default=True,
        help='whether crop face.')
    parser.add_argument('--image_size', type=int, default=224,
        help='input image size.')
    return parser.parse_args()


## functions
def load_and_transform(file):
    img = np.array(Image.open(file).convert("RGB"))
    if args.crop_face:
        faces = detector(img, 0)
        if len(faces) == 0:
            return None
        else:
            face = faces[0]
            img = img[face.top():face.bottom(),face.left():face.right(),:]
    img = Image.fromarray(img).resize((args.image_size,args.image_size))
    img = np.array(img, dtype="float32")
    img = img/127.5 - 1
    return img
    
## base configuration
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
detector = dlib.get_frontal_face_detector()
LABEL_MAP = {
                "gender":{0:"Male",1:"Female"},
                "race":{0:"White",1:"Black",2:"Asian",3:"Indian",4:"Other"}
            }
## load model
with open("config.json") as fi:
    config=json.load(fi)
## define model
features = config.get(args.base_model) if args.feature_aggregate else None
model = build(input_shape=(args.image_size,args.image_size,3), 
              base_model=args.base_model,
              feature_layers=features, 
              train_backbone=False,
              weight_decay=0.0)
model.trainable = False
model.load_weights(args.model)
## read image
flag = True
while flag:
    inputs = input("> input your image: ")
    if inputs == "q":
        flag = False
        continue
    if not os.path.exists(inputs):
        print("[!] %s does not exists!Please retry!")
    image = load_and_transform(inputs)
    Image.fromarray(((image+1)*127.5).astype(np.uint8)).save("test.jpg")
    if image is None:
        print("[！] No face detected! Please retry!")
        continue
    labels = model.predict(image[np.newaxis,...])
    print(labels)
    age = int(labels[0] * 10)
    gender = LABEL_MAP["gender"][int(labels[1].argmax(-1))]
    race = LABEL_MAP["race"][int(labels[2].argmax(-1))]
    print("Result\t age:%s gender:%s race:%s" % (age,gender,race))
    