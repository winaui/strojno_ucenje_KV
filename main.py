import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# tesnorflow i keras
from keras import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img

# micanje warninga
import warnings 
warnings.filterwarnings("ignore")

print('modules loaded')

# Generate data paths with labels
data_dir = "C:/Users/Emzzz/Desktop/su_KV/baza/flower_images"
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

file_path_series = pd.Series(filepaths, name= 'filepaths')
label_path_series = pd.Series(labels, name='labels')
df = pd.concat([file_path_series, label_path_series], axis= 1)

train_df , dummy_df = train_test_split(df ,train_size = 0.8 , shuffle = True ,random_state = 42 )

valid_df , test_df = train_test_split(dummy_df ,test_size= 0.5 , shuffle = True ,random_state = 42)

print(f"The shape of The Train data is: {train_df.shape}")
print(f"The shape of The Validation data is: {valid_df.shape}")
print(f"The shape of The Test data is: {test_df.shape}")

#print(valid_df.head(25))

train_datagen = ImageDataGenerator(rescale=1./255,)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

print("Training Data")
train_ds = train_datagen.flow_from_dataframe(
    train_df,
    x_col = 'filepaths',
    y_col = 'labels',
    target_size = (224, 224),
    batch_size = batch_size,
    class_mode = 'categorical'
)

print("Validation Data")
valid_ds = validation_datagen.flow_from_dataframe(
    valid_df,
    x_col = 'filepaths',
    y_col = 'labels',
    target_size = (224, 224),
    batch_size = batch_size,
    class_mode = 'categorical'
)

print("Test Data")
test_ds = test_datagen.flow_from_dataframe(
    test_df,
    x_col = 'filepaths',
    y_col = 'labels',
    target_size = (224, 224),
    batch_size = batch_size,
    class_mode = 'categorical'
)

count = train_df['labels'].value_counts()

'''
fig, axs = plt.subplots(1, 2, figsize = (12, 6), facecolor = 'white')

palette = sns.color_palette("pastel6")
sns.set_palette(palette)
axs[0].pie(count, labels = count.index, autopct='%1.1f%%', colors = palette, startangle = 140)
axs[0].set_title('Distribucija kategorija')

sns.barplot(x = count.index, y = count.values, ax = axs[1], palette = "pastel6")
axs[1].set_title('Brojanje kategorija')

plt.tight_layout()

plt.show()
'''

class_indices_train = train_ds.class_indices
print("Indeksi za trening skup:", class_indices_train)

'''
gen_dict = train_datagen.class_indices_train
classes = list(gen_dict.keys())
images , labels = next(train_datagen)

plt.figure(figsize= (20,20))

for i in range(16):
    plt.subplot(4,4,i+1)
    image = images[i] / 255
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name = classes[index]
    plt.title(class_name , color = 'brown' , fontsize= 13,weight="bold")
    plt.axis('off')
plt.show()
'''

from keras._tf_keras.keras.applications import DenseNet121
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.applications import ResNet50, Xception
from keras import regularizers
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.losses import BinaryCrossentropy
from keras._tf_keras.keras.initializers import he_normal
from keras._tf_keras.keras.optimizers import Adam, Adamax 

base_model=tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3))

model=Sequential([
    base_model,
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(rate=0.2),
    Dense(265,activation="relu"),
    Dropout(rate=0.5),
    Dense(5,activation="softmax")
    
                 ])
model.compile(Adamax(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

#Originalno je bilo 100 epoha
history = model.fit(train_ds, epochs= 10, shuffle = False,
#                         steps_per_epoch = len(train_ds),
                        validation_data = valid_ds,
#                         validation_steps = len(validation_ds)
                    )

history_dict = history.history

print(history_dict)

epochs = range(1, len(history_dict['loss']) + 1)
results = {
    'epoch': epochs,
    'loss': history_dict['loss'],
    'val_loss': history_dict['val_loss'],
    'accuracy': history_dict['accuracy'],
    'val_accuracy': history_dict['val_accuracy']
}

import pickle

with open('training_history.pkl', 'rb') as f:
    results = pickle.load(f)

text_file_path = "C:/Users/Emzzz/Desktop/su_KV/strojno-ucenje-lv-ovi/training_history.txt"

with open(text_file_path, 'w') as text_file:
    for key, value in results.items():
        text_file.write(f'{key}: {value}\n')

print(f'Results saved to {text_file_path}')


