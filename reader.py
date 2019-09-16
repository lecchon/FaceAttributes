# -*- coding:utf8 -*-


import os
import random
from PIL import Image,ImageEnhance
import cv2
import numpy as np
from keras.utils import to_categorical

class DataLoader:
    
    def __init__(self, file_list, image_size, labels=None, mode="train", argument=True, crop_face=False):
        self.file_list = file_list
        self.image_size = image_size
        self.NUM_FILES = len(self.file_list)
        assert mode in ["train", "val", "test", "inference"]
        self.mode = mode
        self.crop_face = crop_face
        self.exclude_set = set([])
        if self.crop_face:
            self.cropper = cv2.CascadeClassifier('/home/bigdata/anaconda3/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        self.argument = (argument and (self.mode == "train"))
        if mode in ["train", "val", "test"]:
            if labels is None:
                    self.labels = {}
                    for f in self.file_list:
                        file_name = os.path.basename(f)
                        labels_ = file_name.split("_")
                        self.labels[f] = [int(labels_[0])/10,int(labels_[1]),int(labels_[2])]
            else:
                self.labels = labels
            self.label2file = {}
            for k,v in self.labels.items():
                if (int(v[0]),v[1],v[2]) not in self.label2file.keys():
                    self.label2file[(int(v[0]),v[1],v[2])] = [k]
                else:
                    self.label2file[(int(v[0]),v[1],v[2])].append(k)
    
    def transform(self, f):
        img = Image.open(f).convert("RGB")
        if self.crop_face:
            img = self._crop_faces(img)
            if img is None:
                self.exclude_set.add(f)
                return None,None
        if self.argument:
            ## random brightness enhance
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.normalvariate(1,0.02))
            ## random rotation
            img = img.rotate(random.randint(-20,20))
            ## random horizon flip 
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ## random crop
            w_ = int(self.image_size*1.2)
            h_ = int(self.image_size*1.2)
            img = img.resize((w_,h_))
            left = random.randint(0,w_-self.image_size)
            top = random.randint(0,h_-self.image_size)
            right = left + self.image_size
            bottom = top + self.image_size
            img = img.crop([left,top,right,bottom])
        img = img.resize((self.image_size,self.image_size))
        if hasattr(self,"labels"):
            if self.labels is not None:
                lab = self.labels[f]
        else:
            lab = None
        img = np.array(img, dtype="float32")/127.5 - 1.
        return img,lab
    
    def sampling(self, n):
        """
        Sampling from class keys for data balancing.
        """
        lst = []
        keys = random.choices(list(self.label2file.keys()),k=n)
        for k in keys:
            v = random.choice(self.label2file[k])
            if v not in self.exclude_set:
                lst.append(v)
        return lst
    
    def _crop_faces(self, img_in):
        """
        Crop face and support one face returned only.
        """
        img_arr = np.array(img_in)
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        bboxes = self.cropper.detectMultiScale(gray, 2.5, 5)
        if len(bboxes) == 0:
            img_out = None
        elif len(bboxes) > 0:
            idx = (bboxes[:,2] * bboxes[:,3]).argmax()
            x,y,w,h = bboxes[idx,:]
            img_out = img_in.crop((x,y,x+w,y+h))
        else:
            x,y,w,h = bboxes[0,:]
            img_out = img_in.crop((x,y,x+w,y+h))
        return img_out
        
    def flow(self, batch_size=1):
        """
        Output a data generator.
        """
        while True:
            images = []
            labels = []
            if self.mode == "train":
                files = self.sampling(batch_size)
            else:
                files = self.file_list
                random.shuffle(file)
            i = 0
            while len(images) < batch_size:
                if i >= len(files):
                    ## resampling
                    if self.mode == "train":
                        files = self.sampling(batch_size)
                    else:
                        files = self.file_list
                    i = 0
                if len(files) == 0:
                    continue
                image_,label_ = self.transform(files[i])
                i += 1
                if image_ is None:
                    continue
                images.append(image_)
                labels.append(label_)
            labels = np.array(labels)#, dtype="float32")
            labels = np.split(labels, 3, -1)
            yield np.array(images, dtype="float32"), [labels[0], to_categorical(labels[1],2), to_categorical(labels[2],5)]