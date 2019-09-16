# -*- coding:utf8 -*-


import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import dlib
import argparse
import os
import json

cropper = cv2.CascadeClassifier('/home/bigdata/anaconda3/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()

def parse_args():
    parser = argparse.ArgumentParser(description='Test multi-classes classifier')
    parser.add_argument('--model', type=str, 
        default=None, help='exported pb file')
    parser.add_argument('--testset_dir', type=str, 
        default=None, help='directory of test sets.')
    parser.add_argument('--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--crop_face', type=str, default="none",choices=["dlib","opencv","none"],
        help='method of cropping face, `none` for nothing')
    return parser.parse_args()

def load_and_preprocess(file, method):
    img = Image.open(file).convert("RGB")
    img = crop_face(img, method)
    if img is None:
        return None
    img = np.array(img.resize((224,224)))
    img = np.float32(img)/127.5 - 1
    return img


def crop_face(image, method):
    if method == "none":
        return image
    elif method == "dlib":
        img_in = np.array(image)
        faces = detector(img_in, 0)
        if len(faces) == 0:
            return None
        else:
            face = faces[0] # select the first face only
            image = image.crop([face.left(),face.top(),face.right(),face.bottom()])
            return image
    elif method == "opencv":
        img_in = np.array(image.convert("L"))
        bboxes = cropper.detectMultiScale(img_in, 2.5, 5)
        if len(bboxes) == 0:
            return None
        else:
            x,y,w,h = bboxes[0,:]
            image = image.crop((x,y,x+w,y+h))
            return image

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    LABEL_MAP = {
                "gender":{0:"Male",1:"Female"},
                "race":{0:"White",1:"Black",2:"Asian",3:"Indian",4:"Other"}
            }
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with tf.gfile.FastGFile(args.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') # 导入计算图
    ## get tensors
    inputs = sess.graph.get_tensor_by_name("input_1:0")
    age = sess.graph.get_tensor_by_name("branch_age/age/clip_by_value:0")
    gender =sess.graph.get_tensor_by_name("branch_gender/gender/Softmax:0")
    race =sess.graph.get_tensor_by_name("branch_race/race/Softmax:0")
    ## read dir
    if args.testset_dir is not None:
        if not os.path.exists(args.testset_dir):
            print("[!] `%s` doesn't exists! Please retry!" % args.testset_dir)
            sys.exit(1)
        else:
            files_ = [os.path.join(args.testset_dir, i) for i in os.listdir(args.testset_dir) if not i.startswith(".")]
            images = []
            files = []
            for f in files_:
                img = load_and_preprocess(f, args.crop_face)
                if img is not None:
                    files.append(f)
                    images.append(img)
        
            if len(images) == 0:
                print("[!] No avaliable image files found! Please retry!")
                sys.exist(1)
            else:
                images = np.array(images)
                age_,gender_,race_ = sess.run([age,gender,race], feed_dict={inputs:images})
                age_ = np.int32(age_ * 10).flatten().tolist()
                gender_ = gender_.argmax(1).flatten().tolist()
                race_ = race_.argmax(1).flatten().tolist()
                with open("result.csv","w+") as fo:
                    fo.write("file,age,gender,race\n")
                    for f,a,g,r in zip(files,age_,gender_,race_):
                        fo.write("%s,%s,%s,%s\n" % (f, a, LABEL_MAP["gender"][g], LABEL_MAP["race"][r]))
                print("Finish testing! please check the result file `result.csv`.")
    else:
        print("Entering Interaction mode ...")
        flag = True
        while flag:
            input_img = input("> Please input your test image ('q' for exit): ")
            if input_img == "q":
                flag = False
                print("Bye!")
                continue
            else:
                if not os.path.exists(input_img):
                    print("[!] `%s` does not exists!Please retry!" % input_img)
                else:
                    img_ = load_and_preprocess(input_img)
                    age_,gender_,race_ = sess.run([age,gender,race], feed_dict={inputs:img_[np.newaxis,...]})
                    a = np.int32(age_ * 10).flatten().tolist()[0]
                    g = gender_.argmax(1).flatten().tolist()[0]
                    r = race_.argmax(1).flatten().tolist()[0]
                    print("[...] Result for %s:\t Age:%s\t Gender:%s\t Race:%s" % (input_img, a, LABEL_MAP["gender"][g], LABEL_MAP["race"][r]))