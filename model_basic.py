# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 01:55:39 2022

@author: sjurm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 00:49:36 2022

@author: sjurm
"""

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

dir_path = "train"
dir_path2 = "test"
actions = []
#data = np.concatenate([np.load('dataset3/seq_가족_F_1663653962.npy')])

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_path2 = file_path.split("\\")
        file_path3 = file_path2[1].split(".")
        print(file_path3[0])
        
        data = np.concatenate([
            np.load(file_path)],axis=0)
        
print(data)     
print(data.shape)  

for (root2, directories2, files2) in os.walk(dir_path2):
    for file_test in files2:
        file_path_test = os.path.join(root2, file_test)
        file_path2_test = file_path_test.split("\\")
        file_path3_test = file_path2_test[1].split(".")
        action = file_path3_test[0]
        action_split = action[4:]
        action_split2 = action_split.split("_", maxsplit=2)
        #print(action_split2[0])
        
        actions.append(action_split2[0])
        
        data2 = np.concatenate([
            np.load(file_path_test)],axis=0)

data.shape


x_data = data[:, :, :-1]
x_data2 = data2[:,:,:-1]

labels = data[:, 0, -1]
labels2 = data2[:, 0, -1]

print(x_data.shape)
print(labels.shape)


from tensorflow.keras.utils import to_categorical

y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape

y_data2 = to_categorical(labels2, num_classes=len(actions))



from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_data2 = x_data2.astype(np.float32)
y_data2 = y_data2.astype(np.float32)


#x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2021)
x_train = x_data
x_val = x_data2

y_train = y_data
y_val = y_data2

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


x_train.shape[1:3]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# history = model.fit(
#     x_train,
#     y_train,
#     validation_data=(x_val, y_val),
#     epochs=500,
#     callbacks=[
#         ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
#         ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
#     ]
# )



# import matplotlib.pyplot as plt

# fig, loss_ax = plt.subplots(figsize=(16, 10))
# acc_ax = loss_ax.twinx()

# loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper left')

# acc_ax.plot(history .history['acc'], 'b', label='train acc')
# acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')

# plt.show()

# from sklearn.metrics import multilabel_confusion_matrix
# from tensorflow.keras.models import load_model

# model = load_model('models/model.h5')

# y_pred = model.predict(x_val)

# multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))