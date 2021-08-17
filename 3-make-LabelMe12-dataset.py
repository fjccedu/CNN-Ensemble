# -*- encoding: utf-8 -*-

import os
import shutil
import pandas as pd
import numpy as np

# We randomly selected 180 images of each class from the training set
Class_Num = 12
Image_Num = 180

data_path = '/home/LabelMe-12-50k/train/'
img_savepath = 'LabelMe12/'

labelFile = data_path + 'annotation.txt'


def create_sub_dir():
    for c in range(Class_Num):
        dp = img_savepath + str(c)
        if not os.path.exists(dp):
            os.mkdir(dp)        

#

def save_to_class_dir(fnList, classType):
    for f in fnList:
        fn = str(f).zfill(5)
        pre = fn[:2]
        src = data_path + '00' + pre + '/0' + fn + '.jpg'
        trg = img_savepath + str(classType) + '/' + fn + '.jpg'
        shutil.copyfile(src, trg)
    
#    
if __name__ == '__main__':
    
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)
    #
    create_sub_dir()
    
    # 
    df = pd.read_csv(labelFile, sep='\\s+', header=None)
    
    titleName = ['code']
    
    for i in range(Class_Num):
        titleName.append('C'+ str(i)) 
        
    df.columns = titleName
    
    
    for i in range(Class_Num):
        fn = 'C'+ str(i)
        td = df[df[fn]==1]
        file = td['code'].values
        # 
        np.random.shuffle(file)
        # 
        sf = file[:Image_Num]
        save_to_class_dir(sf, i)
    
    #    
    print("ok")