import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import MobileNetV2

import time
from contextlib import contextmanager


# ---------------------------------------------------------------------------------------------
# TIME-COST FUNCTION
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.2f} s".format(title, (time.time() - t0)))
# ---------------------------------------------------------------------------------------------
def image_resize(image, target_size, fill_value=128.0):

    ih, iw    = target_size 
    h,  w, _  = image.shape 

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h) 
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=fill_value)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    return image_paded
# ---------------------------------------------------------------------------------------------

USE_DataAug = True
#USE_DataAug = False


Max_DataAug_Times = 3
PIC_Rotation_Range = 8

FC_NeuronNum = 128
Dropout_Rate = 0.2

USE_DATASET = 'flower'
#USE_DATASET = 'VOC2007'
# USE_DATASET = 'LabelMe12'

Trian_Epochs = 40
Batch_Size = 32


if USE_DATASET == 'flower':
    cfgClassNum = 17
    img_path = '17flowers/'
    logFilePrefix = "log/flower_Baseline_"
    plt_Title_Str = 'Flower-17: '
elif USE_DATASET == 'VOC2007':    
    cfgClassNum = 19
    img_path = 'VOC2007/'
    logFilePrefix = "log/VOC2007_Baseline_"
    plt_Title_Str = 'VOC2007-19: '
else:    
    cfgClassNum = 12
    img_path = 'LabelMe12/'
    logFilePrefix = "log/LabelMe12_Baseline_"
    plt_Title_Str = 'LabelMe-12: '

#

weights_dir = 'weights/'

BaseModel = 'ResNet50V2'
#BaseModel = 'MobileNetV2'
#BaseModel = 'DenseNet201' 


if BaseModel == 'MobileNetV2':
    plt_Title_Str += 'MobileNet_v2'
if BaseModel == 'ResNet50V2':
    plt_Title_Str += 'ResNet-50_v2'
if BaseModel == 'DenseNet201':
    plt_Title_Str += 'DenseNet-201'


Shuffle_Randon_Seed = 8

start_rand_seed = 0
end_rand_seed = 10
K_Repeats = end_rand_seed - start_rand_seed

MAX_Flod_CV = 5

AVG_INDEX = 'weighted'

# ---------------------------------------------------------------------------------------------
def getDirFileNameList(data_path):
    files = []
    lst = os.listdir(data_path)
    for i in range(len(lst)):
        path = os.path.join(data_path, lst[i])
        if os.path.isfile(path):
            files.append(lst[i])

    return files
# ---------------------------------------------------------------------------------------------
def writeTxtFile(file, mode, strInfo):
    # mode='a' append, mode='w' rewrite
    f = open(file, mode)
    f.write(strInfo)
    f.close()
# ---------------------------------------------------------------------------------------------
def get_compiled_model(BaseModel):
    
    ih, iw = 224, 224
    inputShape = (ih, iw, 3)

    if BaseModel == 'ResNet50V2':
        base_model = ResNet50V2(include_top=False, weights=None, input_shape=inputShape)
        base_model.load_weights(weights_dir + r'resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if BaseModel == 'DenseNet201':
        base_model = DenseNet201(include_top=False, weights=None, input_shape=inputShape)
        base_model.load_weights(weights_dir + r'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if BaseModel == 'MobileNetV2':
        base_model = MobileNetV2(include_top=False, weights=None, input_shape=inputShape)
        base_model.load_weights(weights_dir + r'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=inputShape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(FC_NeuronNum, activation='relu')(x)
    x = tf.keras.layers.Dropout(Dropout_Rate)(x)
    outputs = tf.keras.layers.Dense(cfgClassNum, activation='softmax')(x)
    cnn = tf.keras.Model(inputs, outputs)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    
    
    return cnn, base_model

# -----------------------------------------------------------------
def get_DataAugmentation(trainX, trainY):
    maxNum = len(trainY) * Max_DataAug_Times
    x, y = pic_random_augment(trainX, trainY, maxNum)
    
    return x, y
# ---------------------------------------------------------------------------------------------
def pic_random_augment(trainX, trainY, maxNum):
    datagen = ImageDataGenerator(rotation_range=PIC_Rotation_Range, fill_mode='nearest', horizontal_flip=True, vertical_flip=False)
    
    aug_X = []
    aug_Y = []
    
    batches = 0
    for x_batch, y_batch in datagen.flow(trainX, trainY, batch_size=1):
        aug_X.append(x_batch)
        aug_Y.append(y_batch)
        
        batches += 1
        if batches >= maxNum:
            break
    # 
    aug_X = np.array(aug_X)
    aug_Y = np.array(aug_Y)

    return np.squeeze(aug_X), np.squeeze(aug_Y)

# ---------------------------------------------------------------------------------------------    

lstImg = []
lstLable = []
lstImgFile = []

#
with timer('Loading img...'):
    for i in range(cfgClassNum):    
        filePath = img_path + str(i) + '/'
        lstFiles = getDirFileNameList(filePath)
        
        for file in lstFiles:
            img = cv2.imread(filePath + file)
            
            img = image_resize(img, (224, 224), fill_value=0.)        
            lstImg.append(img)
            lstLable.append(i)
            lstImgFile.append(file)

#

imgData = np.array(lstImg)
labels = np.array(lstLable)
imgFiles = np.array(lstImgFile)

imgData, labels, imgFiles= shuffle(imgData, labels, imgFiles, random_state=Shuffle_Randon_Seed)

accuracy = []

titleName = ['rand_seed', 'nFlod', 'acc', 'precision', 'recall', 'f1']

df = pd.DataFrame()

for k in range(start_rand_seed, end_rand_seed):

    rand_seed = k
    kf = KFold(n_splits=MAX_Flod_CV, shuffle=True, random_state=rand_seed)

    for nFlod, (train_index, test_index) in enumerate(kf.split(imgData, labels)):
        trainX = imgData[train_index]
        testX = imgData[test_index]
        trainY = labels[train_index]
        testY = labels[test_index]
        testImgFiles = imgFiles[test_index]
        
        (trainX0, trainValidateX, trainY0, trainValidateY) = train_test_split(trainX, trainY, test_size=0.25, random_state=rand_seed)
        
        if USE_DataAug:
            with timer('Img data augmentation...'):
                trainX, trainY = get_DataAugmentation(trainX, trainY)
        # 
       
        testY = to_categorical(testY, num_classes=cfgClassNum)
        trainValidateY = to_categorical(trainValidateY, num_classes=cfgClassNum)
        
        # ---------------------------------------------------------------------------------------
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        
        
        trainY = to_categorical(trainY, num_classes=cfgClassNum)
        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            cnn, base_model = get_compiled_model(BaseModel)
        #
        # ---------------------------------------------------------------------------------------
        # due to read image data with sequence, need to shuffle first
        shuffled_x, shuffled_y = shuffle(trainX, trainY, random_state=rand_seed)
       
        with timer('Train time:'):
            history = cnn.fit(x=shuffled_x, y=shuffled_y, batch_size=Batch_Size, epochs=Trian_Epochs, validation_data=(trainValidateX, trainValidateY), verbose=0)
            
        #------------------------------------------------------------------------
        output = cnn.predict(testX)
        y_true = np.argmax(testY, axis=1)
        y_pred = np.argmax(output, axis=1)
        #        

        lstScore = []
        lstScore.append(rand_seed)
        lstScore.append(nFlod)

        acc = metrics.accuracy_score(y_true, y_pred)
        acc = round(acc, 4)

        accuracy.append(acc)
        lstScore.append(acc)

        prec = metrics.precision_score(y_true, y_pred, average=AVG_INDEX)
        prec = round(prec, 4)
        lstScore.append(prec)

        rec = metrics.recall_score(y_true, y_pred, average=AVG_INDEX)
        rec = round(rec, 4)
        lstScore.append(rec)

        f1 = metrics.f1_score(y_true, y_pred, average=AVG_INDEX)
        f1 = round(f1, 4)
        lstScore.append(f1)

        errInfo = ''
        for j in range(len(y_pred)):
            if y_pred[j] != y_true[j]:
                errInfo += '{}:{}->{}\n'.format(testImgFiles[j], y_true[j], y_pred[j])
        errInfo += '----------------------------------------------------------------------'
        errFile = logFilePrefix + "_err_" + BaseModel + ".txt"
        writeTxtFile(errFile, 'a', errInfo + '\n')        

        df = df.append([lstScore], ignore_index=True)
        
        epochs = range(len(history.history['accuracy']))
        plt.figure()
    
        plt.plot(epochs,history.history['accuracy'],'r',label='Training accuracy')
        #plt.plot(epochs,history.history['val_accuracy'],'g',label='Validation accuracy')
    
        plt.plot(epochs,history.history['loss'],'b',label='Training loss')
        #plt.plot(epochs,history.history['val_loss'],'k',label='Validation loss')
    
        # plt.title('Traing and Validation accuracy')
        plt.legend(loc="center right")
    
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('accuracy-loss')
        plt.title(plt_Title_Str) 
        # plt.legend(loc="upper right")
        # save image must before call show 
        plt.savefig("log/TL_" + USE_DATASET + "_" + BaseModel + "_" + str(k) + "_" + str(nFlod) + ".png")        
        
        bak_titleName = df.columns
        df.columns = titleName
        dfStr = df.to_string(header=True, index=False)
        df.columns = bak_titleName
    
        cFile = logFilePrefix + "_txt_" + BaseModel + ".txt"
        writeTxtFile(cFile, 'w', dfStr + '\n')        
    # 
# ------------------------------------------------
csvFile = logFilePrefix + BaseModel + ".csv"
df.columns = titleName
df.to_csv(csvFile, index=False, header=True)

print('finish.')    