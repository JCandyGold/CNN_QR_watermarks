
# -*- coding: utf-8 -*-
import tensorflow as tf

# from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np


@tf.keras.saving.register_keras_serializable()
def full_loss(y_true, y_pred):
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
    s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def main():
    model = tf.keras.models.load_model("./test")

    # input_S  Secret
    # input_C  Cover
    input_s = tf.keras.utils.load_img('s.png', target_size=(64, 64))
    input_s = tf.keras.utils.img_to_array(input_s)
    input_s = np.expand_dims(input_s, axis=0)
    print(input_s.shape)
    
    input_c = tf.keras.utils.load_img('c.png', target_size=(64, 64))
    input_c = tf.keras.utils.img_to_array(input_c)
    input_c = np.expand_dims(input_c, axis=0)
    print(input_c.shape)

    decoded = model.predict([input_s, input_c])

    # decoded_C Encoded Cover
    # decoded_S Decoded Secret

    decoded_s, encoded_c = decoded[...,0:3], decoded[...,3:6]
    
    tf.keras.utils.array_to_img(encoded_c[0]).save('encoded_c.png')
    tf.keras.utils.array_to_img(decoded_s[0]).save('decoded_s.png')
    plt.figure()
    plt.imshow(rgb2gray(decoded_s[0]), cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":
    main()