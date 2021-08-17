# -*- encoding: utf-8 -*-

import os
import shutil

data_path = '/home/segment_data/VOCdevkit/VOC2007'

ann_filepath = data_path + '/Annotations/'
img_filepath = data_path + '/JPEGImages/'
main_filepath = data_path + '/ImageSets/Main/'

img_savepath = 'VOC2007/'

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor', 'person']

Class_Num = len(classes)

def create_sub_dir():
    for c in range(Class_Num):
        dp = img_savepath + str(c)
        if not os.path.exists(dp):
            os.mkdir(dp)        

# 
def save_from_mainset():
    
    postfix = ['_train.txt', '_trainval.txt', '_val.txt']
    for c in range(len(classes)):
        for pfx in postfix:
            fn = main_filepath + classes[c] + pfx
            
            file = open(fn)
            fs = file.readlines()
            for ln in fs:
                img = img_filepath + ln[:6] + '.jpg'
                
                if int(ln[6:-1]) > 0:
                    shutil.copy(img, img_savepath + str(c))
                    
            file.close
    
#    

if __name__ == '__main__':
    
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)
    
    create_sub_dir()
    #
    save_from_mainset()
    
    # Delete images with multiple categories
    duplicate_pic = []

    for i in range(Class_Num-2):
        #
        sf = os.listdir(img_savepath + str(i))
        
        for j in range(i+1, Class_Num-1):
            tf = os.listdir(img_savepath + str(j))
            
            inter_set = set(sf).intersection(set(tf))
            
            if len(inter_set) > 0:
                duplicate_pic = duplicate_pic + list(inter_set)
                
    duplicate_pic = list(set(duplicate_pic))
            
    # Delete duplicate images for all directories
    for i in range(Class_Num-1):
        for pic in duplicate_pic:
            tp = img_savepath + str(i) + '/'+ pic
            if os.path.exists(tp):
                os.remove(tp)
             
    # Delete 'person' category           
    shutil.rmtree(img_savepath + str(Class_Num-1)) 
        
    print("ok")
