# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:05:56 2022

@author: gabri
"""
pip install Pillow
pip install graphviz
pip install pydot
pip install scikit-image
pip install tqdm==2.2.3
pip install scikit-learn

import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import datetime

# Librerias para constuir la arquitectura U-Net
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy

from PIL import Image

from tensorflow import keras
import pydot
import graphviz

import pickle


# -----------------------------------------------------------------------------

import sys

def restart_kernel():
    os.execv(sys.executable, ['python'] + sys.argv)

# -----------------------------------------------------------------------------

width_shape, height_shape = 256, 256 # Tamaño de las imagenes de entrada
batch_size = 32 #NUMERO DE LOTES QUE SE VAN TOMANDO DEL TOTAL DEL CONJUNTO 
epochs = 100

HUELLAS_LIST = ["HUELLA_204030", "HUELLA_204031", "HUELLA_205030"]

Xtotal = [] 
Ytotal = []
for HUELLA in HUELLAS_LIST:
    
    data_path_MULTISPECTRAL = ("C:/PFIRE_FOREST_GALI_U_NET/" + HUELLA + "/1_RESULTADOS/MULTISPECTRAL_ENMASK/")   
    data_path_MASK = ("C:/PFIRE_FOREST_GALI_U_NET/" + HUELLA + "/1_RESULTADOS/MASK_FIRE/")  


    # obtenemos una lista con los archivos dentro de cada carpeta
    data_list_train = os.listdir(data_path_MULTISPECTRAL)


    for folder in data_list_train:

        file_list_per_folder = os.listdir(data_path_MULTISPECTRAL + folder)
        
        file_list_per_folder_ = []
        for files in file_list_per_folder:
            if os.path.isfile(os.path.join(data_path_MULTISPECTRAL, folder, files)) and files.endswith('.TIF'):
                file_list_per_folder_ .append(files)
                
    
        for file in tqdm(file_list_per_folder_):
        
            # leemos cada imagen del dataset de entrenamiento y la redimensionamos
            img = imread(data_path_MULTISPECTRAL + folder +'/'+ file)[:,:,:]  
            Xtotal.append(img)
        
            mask = imread(data_path_MASK + folder + '/' + file) 
            Ytotal.append(mask)




#-----------------------------------------------
# --------------------------- PARA IMAGENES JUNTAS -----------------------


width_shape, height_shape = 256, 256 # Tamaño de las imagenes de entrada
batch_size = 32 #NUMERO DE LOTES QUE SE VAN TOMANDO DEL TOTAL DEL CONJUNTO 
epochs = 100


Xtotal = [] 
Ytotal = []
    
data_path_MULTISPECTRAL = ("C:/PFIRE_FOREST_GALI_U_NET/1_CONJUNTO_TODO/MULTISPECTRAL/B_NDVI_NBR")   
data_path_MASK = ("C:/PFIRE_FOREST_GALI_U_NET/1_CONJUNTO_TODO/MASK_TIF")  

# obtenemos una lista con los archivos dentro de cada carpeta
data_list_train = os.listdir(data_path_MULTISPECTRAL)


file_list_per_folder_ = []
    
for files in data_list_train:
    if os.path.isfile(os.path.join(data_path_MULTISPECTRAL, files)) and files.endswith('.TIF'):
        file_list_per_folder_ .append(files)
                

for file in tqdm(file_list_per_folder_):
    # leemos cada imagen del dataset de entrenamiento y la redimensionamos
    img = imread(data_path_MULTISPECTRAL +'/'+ file)[:,:,:]
    Xtotal.append(img)
    
    mask = imread(data_path_MASK + '/' + file[:-4] + '.TIF') 
    Ytotal.append(mask)



#-----------------------------------------------



# SELECT TRAINING SET (80%)
Xtrain = Xtotal[:10150]
Ytrain = Ytotal[:10150]

# SELECT TESTEO SET (20%)
Xtesteo = Xtotal[10150:]
Ytesteo = Ytotal[10150:]



# TRAINING SET (60%)
X_train = np.asarray(Xtrain,dtype=np.uint8)
print('Xtrain:',X_train.shape)
Y_train = np.asarray(Ytrain,dtype= np.bool)
print('Ytrain:',Y_train.shape)

# TEST SET (20%)
X_testeo = np.asarray(Xtesteo,dtype=np.uint8)
print('Xtesteo:',X_testeo.shape)
Y_testeo = np.asarray(Ytesteo,dtype= np.bool)
print('Ytesteo:',Y_testeo.shape)



plt.imshow(X_train[27])
plt.show()
plt.imshow(Y_train[27])
plt.show()



#------
# Definimos la entrada al modelo
Image_input = Input((height_shape, width_shape, 3)) #OJO CON EL NUMERO DE CANALES SE CAMBIA A 8 O 3...
Image_in = Lambda(lambda x: x / 255)(Image_input) 

#contracting path
conv1 = Conv2D(64,  kernel_size = (3, 3), activation='relu', padding='same')(Image_in)
conv1 = Conv2D(64, kernel_size = (3, 3), activation='relu', padding='same')(conv1)
maxp1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv1)

conv2 = Conv2D(128, kernel_size = (3, 3), activation='relu', padding='same')(maxp1)
conv2 = Conv2D(128, kernel_size = (3, 3), activation='relu', padding='same')(conv2)
maxp2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv2)
 
conv3 = Conv2D(256, kernel_size = (3, 3), activation='relu', padding='same')(maxp2)
conv3 = Conv2D(256, kernel_size = (3, 3), activation='relu', padding='same')(conv3)
maxp3 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv3)
 
conv4 = Conv2D(512, kernel_size = (3, 3), activation='relu', padding='same')(maxp3)
conv4 = Conv2D(512, kernel_size = (3, 3), activation='relu', padding='same')(conv4)
maxp4 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(conv4)
maxp4 = Dropout(0.2)(maxp4)
 
#center
conv5 = Conv2D(1024, kernel_size = (3, 3), activation='relu', padding='same')(maxp4)
conv5 = Conv2D(1024, kernel_size = (3, 3), activation='relu', padding='same')(conv5)

#expansive path
up6 = Conv2DTranspose(512, kernel_size = (2, 2), strides = (2, 2), padding = 'same')(conv5)
up6 = concatenate([up6, conv4]) 
conv6 = Conv2D(512, kernel_size = (3, 3), activation='relu', padding='same')(up6) 
conv6 = Conv2D(512, kernel_size = (3, 3), activation='relu', padding='same')(conv6)
 
up7 = Conv2DTranspose(256,  kernel_size = (2, 2), strides = (2, 2), padding='same')(conv6)
up7 = concatenate([up7, conv3])
conv7 = Conv2D(256,  kernel_size = (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(256,  kernel_size = (3, 3), activation='relu', padding='same')(conv7)
 
up8 = Conv2DTranspose(128, kernel_size = (2, 2), strides = (2, 2), padding='same')(conv7)
up8 = concatenate([up8, conv2])
conv8 = Conv2D(128, kernel_size = (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(128, kernel_size = (3, 3), activation='relu', padding='same')(conv8)
 
up9 = Conv2DTranspose(64, kernel_size = (2, 2), strides = (2, 2), padding='same')(conv8)
up9 = concatenate([up9, conv1], axis = 3)
conv9 = Conv2D(64, kernel_size = (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(64, kernel_size = (3, 3), activation='relu', padding='same')(conv9)
 
outputs = Conv2D(1, kernel_size = (1, 1), activation='sigmoid')(conv9)

#-----

from tensorflow.keras import backend as K

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)


#-----
 
from keras.optimizers import Adam

opt = Adam(learning_rate=0.0001, # establecemos la tasa de aprendizaje en 0.01
    epsilon=1e-07) 

model = Model(inputs=[Image_input], outputs=[outputs])
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', iou])
model.summary()
#------

os.chdir("C:/Users/Gabriel/Documents/FINAL_MODEL_UNET/MODEL_BASED_IOU") 
print(os.getcwd())


from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

%load_ext tensorboard

%reload_ext tensorboard

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks_list = [TensorBoard(logdir, histogram_freq=1),
                  EarlyStopping(monitor = 'val_iou',
                                min_delta = 0.0010, #SE MODIFICA PARA MONITORIZAR IOU DE 0.0002 A 0.0010
                                patience = 6,
                                verbose = 1,
                                mode = "max")]

"""
                  ReduceLROnPlateau(monitor='val_loss',
                      factor=0.1,
                      patience=3,
                      verbose=1,
                      mode="min",
                      min_delta=0.0001,
                      min_lr=0)]
    
"""
          
#-----

results = model.fit(X_train, Y_train,
                    validation_split=0.25, #DEL TOTAL DE DATOS ES EL .20 (60/20/20)
                    batch_size=batch_size, 
                    epochs=epochs,
                    shuffle = True,
                    callbacks=[callbacks_list])





# save the model to disk
os.chdir("C:/Users/Gabriel/Documents/FINAL_MODEL_UNET/MODEL_BASED_IOU") 
print(os.getcwd())


filename = 'MODEL_FINAL_ALL_MAS_INDICES_8_BANDS.sav'
pickle.dump(model, open(filename, 'wb'))

# --- 

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test) #rev
print(result)


# ---------------
# EVALUACION DEL MODELO #

evaluateS = model.evaluate(X_testeo, Y_testeo)
evaluateS

import json

with open('evaluacion_model.json', 'w') as jf: 
    json.dump(evaluateS, jf, ensure_ascii=False, indent=2)



#ANALISIS DE LOS RESULTADOS 

historial_entrenamiento = results.history
historial_entrenamiento

with open('historial_entrenamiento.json', 'w') as jf: 
    json.dump(historial_entrenamiento, jf, ensure_ascii=False, indent=2)




acc = results.history['accuracy']
val_acc = results.history['val_accuracy']

loss = results.history['loss']
val_loss = results.history['val_loss']

IOU = results.history['iou']
val_IOU = results.history['val_iou']

epochs_range = range(46)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

plt.plot(epochs_range, IOU, label='Training IoU')
plt.plot(epochs_range, val_IOU, label='Validation IoU')
plt.legend(loc='upper right')
plt.title('Training and Validation IoU')
plt.show()

# prediccion Y TEST 


from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score

os.chdir("C:/Users/Gabriel/Documents/FINAL_MODEL_UNET/MODEL_BASED_IOU/MULTISPECTRAL_6BANDS") 
print(os.getcwd())

# load the model from disk
filename = 'MODEL_FINAL_ALL_6bands.sav'
model = pickle.load(open(filename, 'rb'))


preds = model.predict(X_testeo)

# ------ #
preds[30]
Y_testeo[30]

plt.imshow(Y_testeo[30])
plt.show()

plt.imshow(preds[30])
plt.show()
# ------ #

mask_predict = (preds > 0.5).astype(np.uint8)
a = accuracy_score(Y_testeo.ravel(),mask_predict.ravel())
r = recall_score(Y_testeo.ravel(),mask_predict.ravel())
p = precision_score(Y_testeo.ravel(),mask_predict.ravel())
f = f1_score(Y_testeo.ravel(),mask_predict.ravel())
k = cohen_kappa_score(Y_testeo.ravel(),mask_predict.ravel())

TEST = {"accuracy": a, "precision": p, "recall": r, "f1_score": f, "kappa_coef": k}
print(TEST)

import json

with open('FINAL_TEST_model.json', 'w') as jf: 
    json.dump(TEST, jf, ensure_ascii=False, indent=2)


precision, recall, f1_score, _ = precision_recall_fscore_support(Y_testeo.ravel(), mask_predict.ravel(), average='binary')
print(precision, recall, f1_score, _)



plt.imshow(np.squeeze(preds[30]))
plt.show()

plt.imshow(Y_testeo[30])
plt.show()


plt.imshow(np.squeeze(preds[27]))
plt.show()

plt.imshow(Y_testeo[27])
plt.show()


#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv# 
#=======================================================================================================================================#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
conda install matplotlib
pip install patchify
pip install imagecodecs
pip install h5py

from PIL import Image 
from matplotlib import pyplot as plt
import numpy as np
from patchify import patchify, unpatchify
import h5py 

# ------------------ SEMANTIC SEGMENTATION TESTING AND IDENTIFICATION OF BURNED AREAS -------------------#

os.chdir("C:/Users/Gabriel/Downloads/IMAGEN_INCENDIO_2022/GEE_FINAL_CORRECTED") 
print(os.getcwd())

image = imread('20220731_RGB_R_I.TIF')[:,:,:]  
#plt.imshow(image)

im_array = np.asarray(image,dtype=np.uint8)

patches_256 = patchify(im_array, (256,256,8), step=256) # HAY QUE CAMBIAR LOS CANALES PARA CADA MODEL 

# --- VISUAL VERIFICATION --- #
patch1 = patches_256[2,3,:,:]
plt.imshow(np.squeeze(patch1))
# --------------------------- #

#B_NDVI_NBR / RGB / RGB_NDVI_NBR / MULTISPECTRAL_6BANDS / MULTI_CON_INDICES_8BANDS
# MODEL_FINAL_B_NDVI_NBR / MODEL_FINAL_RGB / MODEL_FINAL_RGB_NDVI_NBR / MODEL_FINAL_ALL_6bands / MODEL_FINAL_ALL_MAS_INDICES_8_BANDS


os.chdir("C:/Users/Gabriel/Documents/FINAL_MODEL_UNET/MODEL_BASED_IOU/MULTI_CON_INDICES_8BANDS") 
print(os.getcwd())

# load the model from disk
filename = 'MODEL_FINAL_ALL_MAS_INDICES_8_BANDS.sav'
model = pickle.load(open(filename, 'rb'))


L = patches_256.shape[0]*256
W = patches_256.shape[1]*256

predictions = np.zeros((L,W, 1), float)
predictions = patchify(predictions, (256,256,1), step=256)

for i in range(patches_256.shape[0]):
    for j in range(patches_256.shape[1]):
        
        preds = model.predict(patches_256[i,j,:,:])
        predictions[i,j,:,:] = preds
        print(i, j)
        
        
# --- VISUAL VERIFICATION --- #
patch1 = patches_256[2,3,:,:]
plt.imshow(np.squeeze(patch1))
     
# --------------------------- #       

# --------------- PIC REARMING ------------------ #


reconstruted_256_BA = unpatchify(predictions,(L,W,1))
reconstruted_256_BA = (reconstruted_256_BA > 0.49).astype(np.uint8)

plt.imshow(np.squeeze(reconstruted_256_BA))


# ---------------------------------------------- #  

pip install rasterio

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

os.chdir("C:/Users/Gabriel/Downloads") 

import tifffile
tifffile.imwrite("imagen.TIF", reconstruted_256_BA)




# ============================================================================ #
# ---- Pic Georeferentiation Process with GDAL ----
restart_kernel()
from osgeo import gdal

# Ruta de la imagen de referencia con coordenadas y sistema de referencia
imagen_referencia = "C:/Users/Gabriel/Downloads/IMAGEN_INCENDIO_2022/GEE_FINAL_CORRECTED/20220731_RGB_R_I.TIF"

# Ruta de la imagen sin coordenadas ni sistema de referencia
imagen_sin_referencia = "C:/Users/Gabriel/Downloads/imagen.TIF"

# Abrir la imagen de referencia
ds_referencia = gdal.Open(imagen_referencia, gdal.GA_ReadOnly)

# Obtener las coordenadas de la imagen de referencia
geotransform = ds_referencia.GetGeoTransform()

# Obtener la proyección de la imagen de referencia
proyeccion = ds_referencia.GetProjection()

# Abrir la imagen sin referencia
ds_sin_referencia = gdal.Open(imagen_sin_referencia, gdal.GA_Update)

# Establecer las coordenadas y la proyección en la imagen sin referencia
ds_sin_referencia.SetGeoTransform(geotransform)
ds_sin_referencia.SetProjection(proyeccion)

# Cerrar los datasets
ds_referencia = None
ds_sin_referencia = None

print("Georreferentiation Completed.")





