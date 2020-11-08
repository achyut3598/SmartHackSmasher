# example of pix2pix gan for satellite to map image-to-image translation
from PIL.Image import Image
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import tensorflow as tf
from matplotlib import pyplot
import sys
import os
from keras import backend as K
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import threading

# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_src_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model



# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1 = data['arr_0']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    return X1


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    X1 = dataset[randint(0, dataset.shape[0], n_samples)]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X1, y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((X.shape[0], patch_shape, patch_shape, 1))
    return X, y


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, filename, n_epochs=1, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # prepare real samples
        x_real, y_real = generate_real_samples(dataset, n_batch, n_patch)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, x_real, n_patch)
        # update discriminator
        d_loss1 = d_model.train_on_batch(x_real, y_real)
        d_loss2 = d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan, y_gan = generate_fake_samples(g_model, x_real, n_patch)
        # update the generator via the discriminator's error
        g_loss, _, _ = gan_model.train_on_batch(x_gan, [y_gan, x_real])
    saveModel(d_model, "d", filename)
    saveModel(g_model, "g", filename)
    saveModel(gan_model, "gan", filename)


def test(d_model, fileToTest, nameOfModel):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    x_real, y_real = generate_real_samples(fileToTest, 1, n_patch)
    d_loss1 = d_model.train_on_batch(x_real, y_real)
    discriminatorValues[nameOfModel] = d_loss1

def saveModel(model, modelType, filename):
    model.save("savedModels\\"+modelType + filename)


def loadModel(filename):
    return tf.keras.models.load_model("savedModels\\"+filename)

# load all images in a directory into memory
def load_images(path, size=(256, 256)):
    src_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + "\\" + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into satellite and map
        pixels = pixels.reshape(256,256,3)
        src_list.append(pixels)
    return asarray(src_list)

def loadImagesAndSaveToFile(dir):
    for i in os.listdir(dir):
        # load dataset
        src_images = load_images(dir + "\\" + i)
        # save as compressed numpy array
        savez_compressed('imageSetsSavedAsNumpyArrays\\' + str(i) + '.npz', src_images)

def setupTraining(epochs, reloadModels):
    for filename in listdir("imageSetsSavedAsNumpyArrays"):
        # load image data
        dataset = load_real_samples("imageSetsSavedAsNumpyArrays\\"+filename)
        if reloadModels:
            # define the models
            d_model =Model()
            g_model=Model()
            gan_model=Model()
            # define the composite model
            for i in os.listdir("savedModels\\"):
                if filename in i:
                    if (i[0] == "d"):
                        d_model=loadModel(i)
                    elif i[0:3] == "gan":
                        gan_model=loadModel(i)
                    else:
                        g_model=loadModel(i)

            # train model
            train(d_model, g_model, gan_model, dataset, filename, epochs)
            del d_model
            del g_model
            del gan_model
            K.clear_session()
        else:
            # define input shape based on the loaded dataset
            image_shape = dataset.shape[1:]
            # define the models
            d_model = define_discriminator(image_shape)
            g_model = define_generator(image_shape)
            # define the composite model
            gan_model = define_gan(g_model, d_model, image_shape)
            # train model
            train(d_model, g_model, gan_model, dataset, filename, epochs)
            del d_model
            del g_model
            del gan_model
            K.clear_session()
            tf.reset_default_graph()

# Global because threading stuff
discriminatorValues = {}

def setupTesting(fileToRate):
    # Grab the fileToRate as a numpy Array
    # load and resize the image
    size = (256, 256)
    pixels = load_img(fileToRate, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    pixels = pixels.reshape(256, 256, 3)
    savez_compressed("fileToBeTested.npz", [pixels])

    dataset = load_real_samples("fileToBeTested.npz")

    # List to store our threads so we can wait for them later
    threads =[]
    # Find each Model and run all of them against the image
    for i in os.listdir("savedModels\\"):
        if(i[0]=="d"):
            t = threading.Thread(target=test, args=(loadModel(i), dataset, i[1:]))
            t.start()
            threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print("THE MOST LIKELY MATCHING IMAGESET IS: ")
    print(min(discriminatorValues.items(), key=lambda x: x[1]))

# The main decides what function to call based on input arguments
# If no valid function is found, alert the user and close
def main(argv):
    print(
        "Args for Train:\n  ['train'] [loadInitialImages (t/f)] ['filepathToNumberedTrainingFolders'] [epochs (int)] [reloadTrainingModels (t/f)] ")
    print("Args for test:\n  [test] [fileToRate]")
    if argv[1] == "train":
        if sys.argv[2] == "t":
            loadImagesAndSaveToFile(str(sys.argv[3]))
        # Initiate Training
        if str(argv[5]) == 't':
            setupTraining(int(str(argv[4])), True)
        else:
            setupTraining(int(str(argv[4])), False)
    elif argv[1] == "test":
        setupTesting(argv[2])

# This is the entry point and only calls main
if __name__ == "__main__":
    main(sys.argv)
