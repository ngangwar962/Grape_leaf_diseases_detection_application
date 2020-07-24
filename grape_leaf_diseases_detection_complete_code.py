#plant datset making connection to kaggle to download the dataset
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 kaggle.json

#plant dataset extracting the data using api from kaggle
!kaggle datasets download -d xabdallahali/plantvillage-dataset
!ls

#unzipping plant dataset 
!unzip -q plantvillage-dataset.zip

#finding number of images present inside the classes
path1="/content/plantvillage dataset/color/Grape___Black_rot/"
path2="/content/plantvillage dataset/color/Grape___Esca_(Black_Measles)/"
path3="/content/plantvillage dataset/color/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/"
path4="/content/plantvillage dataset/color/Grape___healthy/"
count1=0
count2=0
count3=0
count4=0
import os
for im in os.listdir(path1):
  count1+=1
print(count1)
for im in os.listdir(path2):
  count2+=1
print(count2)
for im in os.listdir(path3):
  count3+=1
print(count3)
for im in os.listdir(path4):
  count4+=1
print(count4)
print(count1+count2+count3+count4)

#moving inside the folder
cd plantvillage dataset

#checking the present working directory
pwd

#making folders
os.mkdir("train/")
os.mkdir("test/")


#moving images inside the folders named in train and test 
import glob
import shutil
import os
co1=0
dst_dir_train1 = "train/Black_rot"
dst_dir_test1="test/Black_rot"
dst_dir_train2 = "train/Esca"
dst_dir_test2 ="test/Esca"
dst_dir_train3 = "train/Leaf_Blight"
dst_dir_test3 ="test/Leaf_Blight"
dst_dir_train4 = "train/Healthy"
dst_dir_test4 ="test/Healthy"
os.mkdir(dst_dir_train1)
os.mkdir(dst_dir_test1)
os.mkdir(dst_dir_train2)
os.mkdir(dst_dir_test2)
os.mkdir(dst_dir_train3)
os.mkdir(dst_dir_test3)
os.mkdir(dst_dir_train4)
os.mkdir(dst_dir_test4)
co1=0
co2=0
co3=0
co4=0
for im in os.listdir(path1):
  if(co1<=942):
    shutil.copy(path1+im,dst_dir_train1)
  else:
    shutil.copy(path1+im,dst_dir_test1)
  co1+=1
for im in os.listdir(path2):
  if(co2<=1099):
    shutil.copy(path2+im,dst_dir_train2)
  else:
    shutil.copy(path2+im,dst_dir_test2)
  co2+=1
for im in os.listdir(path3):
  if(co3<=834):
    shutil.copy(path3+im,dst_dir_train3)
  else:
    shutil.copy(path3+im,dst_dir_test3)
  co3+=1
for im in os.listdir(path4):
  if(co4<=334):
    shutil.copy(path4+im,dst_dir_train4)
  else:
    shutil.copy(path4+im,dst_dir_test4)
  co4+=1

#checking the number of images present inside the train dataset folders
count1=0
count2=0
count3=0
count4=0
for im in os.listdir(dst_dir_train1):
  count1+=1
for im in os.listdir(dst_dir_train2):
  count2+=1
for im in os.listdir(dst_dir_train3):
  count3+=1
for im in os.listdir(dst_dir_train4):
  count4+=1
print(count1,count2,count3,count4)
print(count1+count2+count3+count4)

#finding the number of test images present inside the folders
count1=0
count2=0
count3=0
count4=0
for im in os.listdir(dst_dir_test1):
  count1+=1
for im in os.listdir(dst_dir_test2):
  count2+=1
for im in os.listdir(dst_dir_test3):
  count3+=1
for im in os.listdir(dst_dir_test4):
  count4+=1
print(count1,count2,count3,count4)
print(count1+count2+count3+count4)

#final code for training the model and testing over the test dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

#setting image height and width
img_width, img_height = 224, 224

train_data_dir = '/content/plantvillage dataset/train/'
validation_data_dir = '/content/plantvillage dataset/test/'

#giving the number of training and testing samples
nb_train_samples =3213
nb_validation_samples =849

#number of epochs for which the program will run
epochs = 20
#batch size of the assigned images
batch_size = 64
#number of classes in the model
num_classes=4

#checking the image format
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#making the model
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.30))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#feeding the data using train,test and validation generatot
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#saving the model to the local directory
#from keras.models import load_model

model.save_weights('model_saved.h5')