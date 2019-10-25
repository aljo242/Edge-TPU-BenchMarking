from __future__ import print_function
import imageio
from PIL import Image
import numpy as np


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


def create_alexnet(weights_path=None):
    data_format = 'channels_last'
    input = Input(shape=(108, 108, 3))
    conv1_11x11_s4 = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', name='conv1_11x11_s4',
                          kernel_regularizer=l2(0.0002), data_format = 'channels_last')(input)
    pool1_2x2_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1_3x3_s2', data_format ='channels_last')(conv1_11x11_s4)

    conv2_11x11_s1 = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv2_11x11_s1', kernel_regularizer=l2(0.0002), data_format='channels_last')(
        pool1_2x2_s2)
    pool2_2x2_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2_2x2_s2', data_format ='channels_last')(conv2_11x11_s1)

    conv3_3x3_s1 = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3_3x3_s1', data_format='channels_last')(pool2_2x2_s2)
    conv4_3x3_s1 = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4_3x3_s1', data_format='channels_last')(conv3_3x3_s1)
    conv5_3x3_s1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5_3x3_s1', data_format='channels_last')(conv4_3x3_s1)
    pool3_2x2_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3_3x3_s2', data_format ='channels_last')(conv5_3x3_s1)


    # START OUTPUT CLASSIFIER

    classifier_flat = Flatten()(pool3_2x2_s2)
    classifier_fc1 = Dense(4096, activation='relu', name = 'fullyconnected1', kernel_regularizer=l2(.0002))(classifier_flat)
    classifier_act = Dense(3755, activation='softmax', name = 'out', kernel_regularizer=l2(.0002))(classifier_fc1)
    #classifier_fc1 = Dense(4096, activation='relu', name='classifier_fc1', kernel_regularizer=l2(.2))(classifier_flat)
    #classifier_fc2 = Dense(4096, activation='relu', name='classifier_fc2', kernel_regularizer=l2(.2))(classifier_fc1)
    #classifier_out = Dense(3755, activation='softmax', name = 'out', kernel_regularizer=l2(.2))(classifier_fc2)

    alexnet = Model(inputs=input, outputs=classifier_act)
    return alexnet


if __name__ == "__main__":
    img = imageio.imread('cat.jpg', pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((108, 108))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.expand_dims(img, axis=0)


    # Test pretrained model
    HCCR = create_alexnet()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    HCCR.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = HCCR.predict(img)  # note: the model has three outputs

    HCCR.save("hccr_alex_keras.h5")
    print("Saved Model to Disk.")
    new_model = load_model("hccr_alex_keras.h5")
    new_model.summary()


