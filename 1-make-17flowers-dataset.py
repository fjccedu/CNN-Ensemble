# -*- encoding: utf-8 -*-
import os
import shutil

# This dataset consists of 1360 flower images categorized into 17 categories, with 80 images in each category. 
Class_Num = 17
Img_Num = 80


data_path = '/home/17flowers/jpg/'
img_savepath = '17flowers/'


def create_sub_dir():
    for c in range(Class_Num):
        dp = img_savepath + str(c)
        if not os.path.exists(dp):
            os.mkdir(dp)        

#    
if __name__ == '__main__':
    
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)
    
    #
    create_sub_dir()
    
    img_no = 1
    for i in range(Class_Num):
        for j in range(Img_Num):
            img_file = 'image_' + str(img_no).zfill(4) + '.jpg'
            img_src = data_path + img_file
            img_trg = img_savepath + str(i) + '/' + img_file
            
            shutil.copyfile(img_src, img_trg)
            
            img_no += 1
    
    #    
    print("ok")