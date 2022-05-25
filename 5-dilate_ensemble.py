import os
import pandas as pd
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.stats import entropy

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import MobileNetV2

import time
from contextlib import contextmanager


# ------------------------------------------------------------------------------------------------------------------------
# TIME-COST FUNCTION
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.2f} min".format(title, (time.time() - t0) / 60))

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

# ------------------------------------------------------------------------------------------------------------------------
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
# USE_DataAug = False


Max_DataAug_Times = 3
PIC_Rotation_Range = 8

# SUB_IMAGE_MEAN = False
SUB_IMAGE_MEAN = True


USE_BEST_WEIGHT = False
# USE_BEST_WEIGHT = True

Dropout_Rate = 0.2

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

USE_DATASET = 'flower'
#USE_DATASET = 'VOC2007'
#USE_DATASET = 'LabelMe12'

if USE_DATASET == 'flower':
    cfgClassNum = 17
    img_path = '17flowers/'
    logFilePrefix = "log/bFlower_Dilate_"
    filtersNum = 32
    Weighted_Lambda = 0.1
    Trian_Batch_Size = 32
    Trian_Epochs = 40
elif USE_DATASET == 'VOC2007':
    cfgClassNum = 19
    img_path = 'VOC2007/'
    logFilePrefix = "log/bVOC2007_Dilate_"
    filtersNum = 32
    Weighted_Lambda = 0.4
    Trian_Batch_Size = 32
    Trian_Epochs = 40
elif USE_DATASET == 'LabelMe12':    
    cfgClassNum = 12
    img_path = 'LabelMe12/'
    logFilePrefix = "log/bLabelMe12_Dilate_"
    filtersNum = 32
    Weighted_Lambda = 0.8
    Trian_Batch_Size = 32
    Trian_Epochs = 40
    
# 
#USE_Weighted_Ensamble = False
USE_Weighted_Ensamble = True

if USE_Weighted_Ensamble:
    Weighted_Mode = "KLD"


# neurons num in the FC layers 
FC_NeuronNum = 128
Dropout_Rate = 0.2

weights_dir = 'weights/'

#BaseModel = 'ResNet50V2'  
#BaseModel = 'MobileNetV2' 
BaseModel = 'DenseNet201'


Shuffle_Randon_Seed = 8

start_rand_seed = 0
end_rand_seed = 10
K_Repeats = end_rand_seed - start_rand_seed

AVG_INDEX = 'weighted'

MAX_Flod_CV = 5


if USE_DATASET == 'flower':
    lst_dilation_rate = [5, 7]
    # 3,5,9,11,13
    lst_kernel_size = [(i, i) for i in range(3, 6, 2)]
    # skip 7
    for i in range(9, 14, 2):
        lst_kernel_size.append((i, i))
elif USE_DATASET == 'VOC2007':
    lst_dilation_rate = [3, 5, 7]
    # 3,5,7,9,11
    lst_kernel_size = [(i, i) for i in range(3, 12, 2)]
elif USE_DATASET == 'LabelMe12':
    lst_dilation_rate = [3, 5]
    # 3,5,7,9,11
    lst_kernel_size = [(i, i) for i in range(3, 12, 2)]

# ------------------------------------------------------------------------
lst_kernel = []
lst_dilation = []

Classer_Num = 0

for i in lst_kernel_size:
    for j in lst_dilation_rate:
        Classer_Num += 1
        lst_kernel.append(i)
        lst_dilation.append(j)


print('Total individual classer num = ' + str(Classer_Num))
print(lst_kernel)
print(lst_dilation)


# ---------------------------------------------------------------------------------------------
# 
def get_single_compiled_model(BaseModel, kernelSize, dilateSize):
    # 
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
    
    x = tf.keras.layers.Conv2D(filtersNum, kernelSize, padding='same', activation='relu', dilation_rate=dilateSize)(x)    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(FC_NeuronNum, activation='relu', name="fc_out")(x)
    x = tf.keras.layers.Dropout(Dropout_Rate)(x)
    outputs = tf.keras.layers.Dense(cfgClassNum, activation='softmax')(x)
    
    cnn = tf.keras.Model(inputs, outputs)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return cnn, base_model
# 
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
def pic_aug_random(trainX, trainY, trainC, maxNum):
    datagen = ImageDataGenerator(rotation_range=8, fill_mode='nearest', horizontal_flip=True, vertical_flip=False)

    aug_X = []
    aug_Y = []
    aug_C = []
    
    batches = 0
    for x_batch, y_batch in datagen.flow(trainX, trainY, batch_size=1):
        aug_X.append(x_batch)
        aug_Y.append(y_batch)
        #
        aug_C.append(trainC[0])
        
        batches += 1
        if batches >= maxNum:
            break
    # 
    aug_X = np.array(aug_X)
    aug_Y = np.array(aug_Y)
    aug_C = np.array(aug_C)

    return np.squeeze(aug_X), np.squeeze(aug_Y), np.squeeze(aug_C)

# ---------------------------------------------------------------------------------------------

def get_weight_pred(lstCnn, X):
    
    classerNum = len(lstCnn)

    lst_pred = []
    
    for i in range(classerNum):
        pred = lstCnn[i].predict(X)
        y_pred = np.argmax(pred, axis=1)
        lst_pred.append(y_pred)        
    #
    return lst_pred
# --------------------------------------------------
def getKldDiversity(lstPredResult):
    lstKldDiversity = []
    
    for i in range(len(lstPredResult)):
        kld = 0
        p = lstPredResult[i] + 1
        for j in range(len(lstPredResult)):
            if i == j:
                continue
            q = lstPredResult[j] + 1
            kld += entropy(p, q) + entropy(q, p) 
        lstKldDiversity.append(kld)
            
    #
    lstnp = np.array(lstKldDiversity)
    
    if sum(lstnp) == 0:
        return 0
    else:
        return lstnp/sum(lstnp)
        
# ----------------------------------------------------
def get_model_weights(y_true, lstPred, lda):
    
    classerNum = len(lstPred)
    
    lstAcc = []
    
    for i in range(classerNum):
        y_pred = lstPred[i]
        acc = metrics.accuracy_score(y_true, y_pred)
        lstAcc.append(acc)        
        
    #
    npAcc = np.array(lstAcc)
    w = npAcc / sum(npAcc)
    
    if Weighted_Mode == "KLD":
        kld = getKldDiversity(lstPred)
        w = w * (1 - lda) + kld * lda
    
    # 
    return w
# ---------------------------------------------------
def get_w_avg_std(trainWeightY, lstWeightPred):
    # 
    classerNum = len(lstWeightPred)
    
    lstAcc = []
    
    for i in range(classerNum):
        y_pred = lstWeightPred[i]
        acc = metrics.accuracy_score(trainWeightY, y_pred)
        lstAcc.append(acc)        
        
    #
    raw_acc = np.array(lstAcc)
    avg_acc = np.mean(raw_acc)
    acc_std = np.std(raw_acc, ddof=1)
    
    return round(avg_acc, 4), round(acc_std, 4), round(np.max(raw_acc), 4), round(np.min(raw_acc), 4)


# ---------------------------------------------------
def test_Weighted_Lambda(trainWeightY, lstValidatePred, lstPred, y_true):
    #
    best_lamda = 0
    best_lamda_str = ''
    best_acc = 0
    for d in range(0, 10):
        lmd = d / 10.0
        tw = get_model_weights(trainWeightY, lstValidatePred, lmd)
        
        lst_pred = []
        for w in range(len(tw)):
            lst_pred.append(lstPred[w] * tw[w])
        #    
        lst_pred = np.array(lst_pred)
        
        output = np.mean(lst_pred, axis=0)
        y_pred = np.argmax(output, axis=1)
        
        acc = metrics.accuracy_score(y_true, y_pred)
        print(d, acc)
        
        if acc == best_acc:
            best_lamda_str += ',' + str(lmd)
	    
        if acc > best_acc:
            best_acc = acc
            best_lamda = lmd
            best_lamda_str = str(best_lamda)
    
    #
    return best_lamda_str, round(best_acc, 4)
# -------------------------------------------------------   
def get_multi_out_model_weights(cnn, X, Y):
    
    lstPred = cnn.predict(X)
    lstPred = np.array(lstPred)
    
    y_true = np.argmax(Y, axis=1)
    
    # 
    classerNum = lstPred.shape[0]
    lstAcc = []
    for i in range(classerNum):
        y_pred = np.argmax(lstPred[i], axis=1)
        acc = metrics.accuracy_score(y_true, y_pred)
        lstAcc.append(acc)
        
    #
    lstAcc = np.array(lstAcc)
    amin, amax = lstAcc.min(), lstAcc.max() 
    w = (lstAcc-amin)/(amax-amin) 
    
    return w
# ---------------------------------------------------------------------------------------------
def addLostClass(tx, ty, rawX, rawY):
    # 
    n_class = len(np.unique(rawY))        
    label_id = [label for label in range(n_class)]
        
    label_train = np.unique(ty)
    label_lost = np.setdiff1d(label_id, label_train)
    
    for label in label_lost:
        print('Lost class:' + str(label))
        label_idx = np.where(trainY==label)[0][0]
        add_X_Train = rawX[label_idx]
        add_Y_Train = rawY[label_idx]
        add_X_Train = np.expand_dims(add_X_Train, axis=0)
        tx = np.vstack((tx, add_X_Train))
        ty = np.append(ty, add_Y_Train)
    
    return tx, ty
# ----------------------------------------------------------------------------------------------
def chkIncludeFullClass(tx, vx, ty, vy, trainX, trainY):
    
    tx, ty = addLostClass(tx, ty, trainX, trainY)
    vx, vy = addLostClass(vx, vy, trainX, trainY)
    
    return tx, vx, ty, vy

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

titleName = ['seed', 'nFlod', 'avg_acc', 'acc_std', 'v_acc', 'v_precision', 'v_recall', 'v_f1', 'w_acc', 'w_precision', 'w_recall', 'w_f1', 'best_lamda', 'best_w_acc', 'use_lamda', 'w_avg_acc', 'w_acc_std', 'w_acc_max', 'w_acc_min', 'w_gap_rate']

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
	
        if USE_Weighted_Ensamble:
            (trainValidateX, trainWeightX, trainValidateY, trainWeightY) = train_test_split(trainValidateX, trainValidateY, test_size=0.25, random_state=rand_seed)
        
        trainX, trainValidateX, trainY, trainValidateY = chkIncludeFullClass(trainX0, trainValidateX, trainY0, trainValidateY, trainX, trainY)
        
        if USE_DataAug:
            with timer('Img data augmentation...'):
                trainX, trainY = get_DataAugmentation(trainX, trainY)
        #
        
        if SUB_IMAGE_MEAN:
            lstMean = []
            for i in range(3):
                lstMean.append(trainX[:, :, :, i].mean())
            
            for i in range(3):
                trainX[:, :, :, i] -= lstMean[i]
                trainValidateX[:, :, :, i] -= lstMean[i]
                testX[:, :, :, i] -= lstMean[i]
        #
        
        testY = to_categorical(testY, num_classes=cfgClassNum)
        trainValidateY = to_categorical(trainValidateY, num_classes=cfgClassNum)
        
        # ---------------------------------------------------------------------------------------
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        
        trainY = to_categorical(trainY, num_classes=cfgClassNum)
        
        lstCnn = []
        lstBase_model = []
        for c in range(Classer_Num):
            #
            print("training classer: " + str(c))
            
            # Open a strategy scope.
            with strategy.scope():
                # Everything that creates variables should be under the strategy scope.
                # In general this is only model construction & `compile()`.
                cnn, base_model = get_single_compiled_model(BaseModel, lst_kernel[c], lst_dilation[c])
            #
            
            shuffled_x, shuffled_y = shuffle(trainX, trainY, random_state=rand_seed)
   
            with timer('Train time:'):
                cnn.fit(x=shuffled_x, y=shuffled_y, batch_size=Trian_Batch_Size, epochs=Trian_Epochs, validation_data=(trainValidateX, trainValidateY), callbacks=[callback], verbose=0)
                
            #
            lstCnn.append(cnn)
            lstBase_model.append(base_model)
        #
        #------------------------------------------------------------------------
        y_true = np.argmax(testY, axis=1)
        
        lstPred = []
        raw_acc = []
        for c in range(Classer_Num):
            pred = lstCnn[c].predict(testX)
            lstPred.append(pred)
            
            raw_y_pred = np.argmax(pred, axis=1)
            tacc = metrics.accuracy_score(y_true, raw_y_pred)
            tacc = round(tacc, 4)
            raw_acc.append(tacc)
            
        #
        lstPred = np.array(lstPred)
        
        raw_acc = np.array(raw_acc)
        avg_acc = np.mean(raw_acc)
        acc_std = np.std(raw_acc, ddof=1)
        
        # 
        lstScore = []
        lstScore.append(rand_seed)
        lstScore.append(nFlod)
        lstScore.append(avg_acc)
        lstScore.append(acc_std)
        

        output = np.mean(lstPred, axis=0)
        y_pred = np.argmax(output, axis=1)

        acc = metrics.accuracy_score(y_true, y_pred)
        acc = round(acc, 4)
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
        
        lstWeightPred = get_weight_pred(lstCnn, trainWeightX)
	
        w_avg_acc, w_acc_std, w_acc_max, w_acc_min = get_w_avg_std(trainWeightY, lstWeightPred)
        w_gap_rate = round((w_acc_max - w_acc_min) / w_avg_acc, 4)
	
        best_lamda, best_w_acc = test_Weighted_Lambda(trainWeightY, lstWeightPred, lstPred, y_true)
	    
        lstWeight = get_model_weights(trainWeightY, lstWeightPred, Weighted_Lambda)
        for w in range(len(lstWeight)):
            lstPred[w] *= lstWeight[w]
            
        
        output = np.mean(lstPred, axis=0)
        y_pred = np.argmax(output, axis=1)

        acc = metrics.accuracy_score(y_true, y_pred)
        acc = round(acc, 4)

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
        
        lstScore.append(best_lamda)
        lstScore.append(best_w_acc)
        lstScore.append(Weighted_Lambda)
        
        lstScore.append(w_avg_acc)
        lstScore.append(w_acc_std)
        lstScore.append(w_acc_max)
        lstScore.append(w_acc_min)
        lstScore.append(w_gap_rate)
        
        #-------------------------------------------------------------
        errFile = testImgFiles[y_pred != y_true]

        errInfo = ''
        for j in range(len(y_pred)):
            if y_pred[j] != y_true[j]:
                errInfo += '{}:{}->{}\n'.format(testImgFiles[j], y_true[j], y_pred[j])
        errInfo += '----------------------------------------------------------------------'
        errFile = logFilePrefix + "_err_" + BaseModel + ".txt"
        writeTxtFile(errFile, 'a', errInfo + '\n')
        #
        df = df.append([lstScore], ignore_index=True)

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
