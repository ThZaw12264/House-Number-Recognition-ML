import initdata as id
import gradcam as gc
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from keras import Sequential
from keras import layers

tf.keras.utils.set_random_seed(100101)
im_trx, im_testx, im_try, im_testy = id.cnndata()

# Model for GRAD-CAM
model = Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32,32,3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.GlobalAveragePooling2D()) 
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(im_trx, im_try, epochs=1,validation_data=(im_testx, im_testy))

pred_test = model.predict(im_testx)
indexes = tf.argmax(pred_test, axis=1)

fig, ax = plt.subplots(5, 2)
fig.set_size_inches(20,20)
for i in range(5):
    icam = gc.GradCAM(model, indexes[i], 'conv2d_2')
    image = tf.reshape(im_testx[i], [1, 32, 32, 3])
    heatmap = icam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (32, 32))
    image = cv2.resize(im_testx[i], (32, 32))
    ax[i,0].imshow(heatmap)
    ax[i,1].imshow(image)
    ax[i,0].axis('off')
    ax[i,1].axis('off')
plt.show()


# Best model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = model.fit(im_trx, im_try, epochs=1,validation_data=(im_testx, im_testy))
pred_test = model.predict(im_testx)
indexes = tf.argmax(pred_test, axis=1)

fig, ax = plt.subplots(6, 6, figsize=(16,16))

for i in range(6):
    for j in range(6):
        ax[i,j].imshow(im_testx[i*7+j])
        ax[i,j].axis('off')
        ax[i,j].set_title(f'P: {(1+indexes[i*7+j])%10}, A: {(1+im_testy[i*7+j,0])%10}', size=10)

plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend(loc='lower right')
plt.show()