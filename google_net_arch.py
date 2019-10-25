sfrom __future__ import print_function
import imageio
from PIL import Image
import numpy as np
import keras

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    Concatenate, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.layers.core import Layer
# from lrn import LRN
from keras.models import load_model

if keras.backend.backend() == 'tensorflow':
    from tensorflow.keras import backend as K
    import tensorflow as tf
    from keras.utils.conv_utils import convert_kernel



def create_googlenet(weights_path=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    data_format = 'channels_last'
    input = Input(shape=(120, 120, 3))
    conv1_7x7_s2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1/7x7_s2',
                          kernel_regularizer=l2(0.0002), data_format = 'channels_last')(input)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1/3x3_s2', data_format = 'channels_last')(conv1_7x7_s2)
    #pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

    conv2_3x3_reduce = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2/3x3_reduce',
                              kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool1_3x3_s2)
    conv2_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(
        conv2_3x3_reduce)
    #conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2/3x3_s2', data_format = 'channels_last')(conv2_3x3)

    # Start PARALLEL LEVEL 1
    inception_3a_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_3a/1x1',
                              kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool2_3x3_s2)

    inception_3a_3x3_reduce = Conv2D(96, (1, 1), padding='same', activation='relu', name='inception_3a/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool2_3x3_s2)
    inception_3a_3x3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='inception_3a/3x3',
                              kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3a_3x3_reduce)

    inception_3a_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu', name='inception_3a/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool2_3x3_s2)
    inception_3a_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu', name='inception_3a/5x5',
                              kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3a_5x5_reduce)

    inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a/pool', data_format = 'channels_last')(
        pool2_3x3_s2)
    inception_3a_pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu', name='inception_3a/pool_proj',
                                    kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3a_pool)

    inception_3a_output = Concatenate(axis=3, name='inception_3a/output')(
        [inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj])

    # START INCEPTION MODULE 2
    inception_3b_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_3b/1x1',
                              kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3a_output)

    inception_3b_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_3b/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3a_output)
    inception_3b_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='inception_3b/3x3',
                              kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3b_3x3_reduce)

    inception_3b_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='inception_3b/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3a_output)
    inception_3b_5x5 = Conv2D(96, (5, 5), padding='same', activation='relu', name='inception_3b/5x5',
                              kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3b_5x5_reduce)

    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b/pool', data_format = 'channels_last')(
        inception_3a_output)
    inception_3b_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_3b/pool_proj',
                                    kernel_regularizer=l2(0.0002), data_format = 'channels_last')(inception_3b_pool)

    inception_3b_output = Concatenate(axis=3, name='inception_3b/output')(
        [inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj])
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3/3x3_s2', data_format = 'channels_last')(inception_3b_output)

    # START INCEPTION MODULE 3
    icp3_out0 = Conv2D(160, (1, 1), padding='same', activation='relu', name='icp3_out0',
                       kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool3_3x3_s2)

    icp3_reduction1 = Conv2D(112, (1, 1), padding='same', activation='relu', name='icp3_reduction1',
                             kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool3_3x3_s2)
    icp3_out1 = Conv2D(224, (3, 3), padding='same', activation='relu', name='icp3_out1',
                       kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp3_reduction1)

    icp3_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu', name='icp3_reduction2',
                             kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool3_3x3_s2)
    icp3_out2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='icp3_out2',
                       kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp3_reduction2)

    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool3', data_format = 'channels_last')(pool3_3x3_s2)
    icp3_out3 = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(pool3)

    icp3_output = Concatenate(axis=3, name='icp3_output')(
        [icp3_out0, icp3_out1, icp3_out2, icp3_out3])

    # START INCEPTION MODULE 4
    icp4_out0 = Conv2D(256, (1, 1), padding='same', activation='relu', name='icp4_out0', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp3_output)

    icp4_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu', name='icp4_reduction_pad', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp3_output)
    icp4_out1 = Conv2D(320, (3, 3), padding='same', activation='relu', name='icp4_out1', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp4_reduction1)

    icp4_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp4_reduction2', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp3_output)
    icp4_out2 = Conv2D(128, (5, 5), padding='same', activation='relu', name='icp4_out2', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp4_reduction2)

    icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool', data_format = 'channels_last')(
        icp3_output)
    icp4_out3 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp3_out4', kernel_regularizer=l2(0.0002), data_format = 'channels_last')(icp4_pool)

    icp4_output = Concatenate(axis=3, name='icp4_output')(
        [icp4_out0, icp4_out1, icp4_out2, icp4_out3])

    # START OUTPUT CLASSIFIER
    classifier_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name='classifier_pool', data_format = 'channels_last')(icp4_output)
    classifier_conv = Conv2D(128, (1, 1), padding='same', activation='relu', name='classifier_conv',
                             kernel_regularizer=l2(.0002), data_format = 'channels_last')(classifier_pool)
    classifier_flat = Flatten()(classifier_conv)
    classifier_fc = Dense(1024, activation='relu', name = 'fullyconnected', kernel_regularizer=l2(.0002))(classifier_flat)

    classifier_act = Dense(3755, activation='softmax', name = 'out', kernel_regularizer=l2(.0002))(classifier_fc)

    googlenet = Model(inputs=input, outputs=classifier_act)


    return googlenet


if __name__ == "__main__":
    img = imageio.imread('cat.jpg', pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((120, 120))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.expand_dims(img, axis=0)

    # Test pretrained model
    HCCR = create_googlenet()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    HCCR.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = HCCR.predict(img)  # note: the model has three outputs

    HCCR.save("hccr_googlenet_keras.h5")
    print("Saved Model to Disk.")
    new_model = load_model("hccr_googlenet_keras.h5")
    new_model.summary()

"""
    # converting currently with OPTIMIZE FOR SIZE argument
    # With snippets of sample data, will be able to call a full model (weights and other ops) quantization
    converter = tf.lite.TFLiteConverter.from_keras_model_file("hccr_keras.h5",  custom_objects={'LRN': LRN,"PoolHelper": PoolHelper})
    converter.optimizations  = [tf.lite.Optimize.DEFAULT]
    print("Converting and Quantizing...")
    tflite_model = converter.convert()

    print("Saving tflite file...")
    open("hccr_lite.tflite", "wb").write(tflite_model)
"""
