import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.activations import relu

def small_test():
    inputs = Input(shape=(144, 256, 6))

    conv1 = Conv2D(32, 3, padding="same", activation='relu')(inputs)
    conv1 = Conv2D(32, 3, padding="same", activation='relu')(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, padding="same", activation='relu')(maxp1)
    conv2 = Conv2D(64, 3, padding="same", activation='relu')(conv2)

    conc3 = concatenate([UpSampling2D(size=(2, 2))(conv2),conv1])
    conv3 = Conv2D(32, 3, padding="same", activation='relu')(conc3)
    conv3 = Conv2D(32, 3, padding="same", activation='relu')(conv3)

    outputs = Conv2D(3, 1, activation='sigmoid')(conv3)

    return Model(inputs=inputs, outputs=outputs, name='small_test')

def u_net():
    inputs = Input(shape=(144, 256, 6))

    conv1 = Conv2D(32, 3, padding="same", activation='relu')(inputs)
    conv1 = Conv2D(32, 3, padding="same", activation='relu')(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, padding="same", activation='relu')(maxp1)
    conv2 = Conv2D(64, 3, padding="same", activation='relu')(conv2)
    maxp2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, padding="same", activation='relu')(maxp2)
    conv3 = Conv2D(128, 3, padding="same", activation='relu')(conv3)
    maxp3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, padding="same", activation='relu')(maxp3)
    conv4 = Conv2D(256, 3, padding="same", activation='relu')(conv4)
    maxp4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, padding="same", activation='relu')(maxp4)
    conv5 = Conv2D(512, 3, padding="same", activation='relu')(conv5)

    conc6 = concatenate([UpSampling2D(size=(2, 2))(conv5),conv4])
    conv6 = Conv2D(256, 3, padding="same", activation='relu')(conc6)
    conv6 = Conv2D(256, 3, padding="same", activation='relu')(conv6)

    conc7 = concatenate([UpSampling2D(size=(2, 2))(conv6),conv3])
    conv7 = Conv2D(128, 3, padding="same", activation='relu')(conc7)
    conv7 = Conv2D(128, 3, padding="same", activation='relu')(conv7)

    conc8 = concatenate([UpSampling2D(size=(2, 2))(conv7),conv2])
    conv8 = Conv2D(64, 3, padding="same", activation='relu')(conc8)
    conv8 = Conv2D(64, 3, padding="same", activation='relu')(conv8)

    conc9 = concatenate([UpSampling2D(size=(2, 2))(conv8),conv1])
    conv9 = Conv2D(32, 3, padding="same", activation='relu')(conc9)
    conv9 = Conv2D(32, 3, padding="same", activation='relu')(conv9)

    outputs = Conv2D(3, 1, activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=outputs, name='u-net')


def u_net_bn():
    inputs = Input(shape=(144, 256, 6))

    conv1 = Conv2D(64, 3, padding="same", activation='relu')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, padding="same", activation='relu')(bn1)
    bn1 = BatchNormalization()(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, 3, padding="same", activation='relu')(maxp1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, padding="same", activation='relu')(bn2)
    bn2 = BatchNormalization()(conv2)
    maxp2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, 3, padding="same", activation='relu')(maxp2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, padding="same", activation='relu')(bn3)
    bn3 = BatchNormalization()(conv3)
    maxp3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, 3, padding="same", activation='relu')(maxp3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, padding="same", activation='relu')(bn4)
    bn4 = BatchNormalization()(conv4)
    maxp4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(1024, 3, padding="same", activation='relu')(maxp4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, padding="same", activation='relu')(bn5)
    bn5 = BatchNormalization()(conv5)
    dp5 = Dropout(0.5)(bn5)

    conc6 = concatenate([UpSampling2D(size=(2, 2))(dp5),bn4])
    conv6 = Conv2D(512, 3, padding="same", activation='relu')(conc6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, padding="same", activation='relu')(bn6)
    bn6 = BatchNormalization()(conv6)

    conc7 = concatenate([UpSampling2D(size=(2, 2))(bn6),bn3])
    conv7 = Conv2D(256, 3, padding="same", activation='relu')(conc7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, padding="same", activation='relu')(bn7)
    bn7 = BatchNormalization()(conv7)

    conc8 = concatenate([UpSampling2D(size=(2, 2))(bn7),bn2])
    conv8 = Conv2D(128, 3, padding="same", activation='relu')(conc8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, padding="same", activation='relu')(bn8)
    bn8 = BatchNormalization()(conv8)

    conc9 = concatenate([UpSampling2D(size=(2, 2))(bn8),bn1])
    conv9 = Conv2D(64, 3, padding="same", activation='relu')(conc9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, padding="same", activation='relu')(bn9)
    bn9 = BatchNormalization()(conv9)

    outputs = Conv2D(3, 1, activation='sigmoid')(bn9)

    return Model(inputs=inputs, outputs=outputs, name='u-net-bn')

class DCGAN(keras.Model):
    """
    Code inspired by: 
        https://keras.io/guides/customizing_what_happens_in_fit/
        https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
    """
    def __init__(self):
        super(DCGAN, self).__init__()
        self.doptimizer = keras.optimizers.Adam(learning_rate = 1e-5)
        self.goptimizer = keras.optimizers.Adam(learning_rate = 1e-5)
        self.loss_fn = keras.losses.BinaryCrossentropy()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile()
        self.generator = self.build_generator()


        z = Input(shape=(144, 256, 6))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile()



    def build_discriminator(self):        
        inputs = Input(shape=(144, 256, 3))

        conv1 = Conv2D(32, 3, padding="same")(inputs)
        lr1 = LeakyReLU(alpha=0.2)(conv1)
        bn1 = BatchNormalization()(lr1)
        conv1 = Conv2D(32, 3, padding="same")(bn1)
        lr1 = LeakyReLU(alpha=0.2)(conv1)
        bn1 = BatchNormalization()(lr1)
        maxp1 = AveragePooling2D(pool_size=(2, 2))(bn1)

        conv2 = Conv2D(64, 3, padding="same")(maxp1)
        lr2 = LeakyReLU(alpha=0.2)(conv2)
        bn2 = BatchNormalization()(lr2)
        conv2 = Conv2D(64, 3, padding="same")(bn2)
        lr2 = LeakyReLU(alpha=0.2)(conv2)
        bn2 = BatchNormalization()(lr2)
        maxp2 = AveragePooling2D(pool_size=(2, 2))(bn2)

        conv3 = Conv2D(128, 3, padding="same")(maxp2)
        lr3 = LeakyReLU(alpha=0.2)(conv3)
        bn3 = BatchNormalization()(lr3)
        conv3 = Conv2D(128, 3, padding="same")(bn3)
        lr3 = LeakyReLU(alpha=0.2)(conv3)
        bn3 = BatchNormalization()(lr3)
        maxp3 = AveragePooling2D(pool_size=(2, 2))(bn3)

        conv4 = Conv2D(256, 3, padding="same")(maxp3)
        lr4 = LeakyReLU(alpha=0.2)(conv4)
        bn4 = BatchNormalization()(lr4)
        conv4 = Conv2D(256, 3, padding="same")(bn4)
        lr4 = LeakyReLU(alpha=0.2)(conv4)
        bn4 = BatchNormalization()(lr4)
        maxp4 = AveragePooling2D(pool_size=(2, 2))(bn4)

        conv5 = Conv2D(512, 3, padding="same")(maxp4)
        lr5 = LeakyReLU(alpha=0.2)(conv5)
        bn5 = BatchNormalization()(lr5)
        conv5 = Conv2D(512, 3, padding="same")(bn5)
        lr5 = LeakyReLU(alpha=0.2)(conv5)
        bn5 = BatchNormalization()(lr5)
        dp5 = Dropout(0.25)(bn5)
        f5 = Flatten()(dp5)

        d = Dense(64)(f5)
        lr6 = LeakyReLU(alpha=0.2)(d)
        dp6 = Dropout(0.5)(lr6)
        outputs = Dense(1, activation='sigmoid')(dp6)

        return Model(inputs=inputs, outputs=outputs, name='gan-cnn-classifier')

    def build_generator(self):
        inputs = Input(shape=(144, 256, 6))

        conv1 = Conv2D(32, 3, padding="same")(inputs)
        lr1 = LeakyReLU(alpha=0.2)(conv1)
        bn1 = BatchNormalization()(lr1)
        conv1 = Conv2D(32, 3, padding="same")(bn1)
        lr1 = LeakyReLU(alpha=0.2)(conv1)
        bn1 = BatchNormalization()(lr1)
        maxp1 = AveragePooling2D(pool_size=(2, 2))(bn1)

        conv2 = Conv2D(64, 3, padding="same")(maxp1)
        lr2 = LeakyReLU(alpha=0.2)(conv2)
        bn2 = BatchNormalization()(lr2)
        conv2 = Conv2D(64, 3, padding="same")(bn2)
        lr2 = LeakyReLU(alpha=0.2)(conv2)
        bn2 = BatchNormalization()(lr2)
        maxp2 = AveragePooling2D(pool_size=(2, 2))(bn2)

        conv3 = Conv2D(128, 3, padding="same")(maxp2)
        lr3 = LeakyReLU(alpha=0.2)(conv3)
        bn3 = BatchNormalization()(lr3)
        conv3 = Conv2D(128, 3, padding="same")(bn3)
        lr3 = LeakyReLU(alpha=0.2)(conv3)
        bn3 = BatchNormalization()(lr3)
        maxp3 = AveragePooling2D(pool_size=(2, 2))(bn3)

        conv4 = Conv2D(256, 3, padding="same")(maxp3)
        lr4 = LeakyReLU(alpha=0.2)(conv4)
        bn4 = BatchNormalization()(lr4)
        conv4 = Conv2D(256, 3, padding="same")(bn4)
        lr4 = LeakyReLU(alpha=0.2)(conv4)
        bn4 = BatchNormalization()(lr4)
        maxp4 = AveragePooling2D(pool_size=(2, 2))(bn4)

        conv5 = Conv2D(512, 3, padding="same")(maxp4)
        lr5 = LeakyReLU(alpha=0.2)(conv5)
        bn5 = BatchNormalization()(lr5)
        conv5 = Conv2D(512, 3, padding="same")(bn5)
        lr5 = LeakyReLU(alpha=0.2)(conv5)
        bn5 = BatchNormalization()(lr5)
        dp5 = Dropout(0.5)(bn5)

        conc6 = concatenate([UpSampling2D(size=(2, 2))(dp5),bn4])
        conv6 = Conv2D(256, 3, padding="same")(conc6)
        lr6 = LeakyReLU(alpha=0.2)(conv6)
        bn6 = BatchNormalization()(lr6)
        conv6 = Conv2D(256, 3, padding="same")(bn6)
        lr1 = LeakyReLU(alpha=0.2)(conv6)
        bn6 = BatchNormalization()(lr6)

        conc7 = concatenate([UpSampling2D(size=(2, 2))(bn6),bn3])
        conv7 = Conv2D(128, 3, padding="same")(conc7)
        lr7 = LeakyReLU(alpha=0.2)(conv7)
        bn7 = BatchNormalization()(lr7)
        conv7 = Conv2D(128, 3, padding="same")(bn7)
        lr7 = LeakyReLU(alpha=0.2)(conv7)
        bn7 = BatchNormalization()(lr7)

        conc8 = concatenate([UpSampling2D(size=(2, 2))(bn7),bn2])
        conv8 = Conv2D(64, 3, padding="same")(conc8)
        lr8 = LeakyReLU(alpha=0.2)(conv8)
        bn8 = BatchNormalization()(lr8)
        conv8 = Conv2D(64, 3, padding="same")(bn8)
        lr8 = LeakyReLU(alpha=0.2)(conv8)
        bn8 = BatchNormalization()(lr8)

        conc9 = concatenate([UpSampling2D(size=(2, 2))(bn8),bn1])
        conv9 = Conv2D(32, 3, padding="same")(conc9)
        lr9 = LeakyReLU(alpha=0.2)(conv9)
        bn9 = BatchNormalization()(lr9)
        conv9 = Conv2D(32, 3, padding="same")(bn9)
        lr9 = LeakyReLU(alpha=0.2)(conv9)
        bn9 = BatchNormalization()(lr9)

        outputs = Conv2D(3, 1, activation='tanh')(bn9)

        return Model(inputs=inputs, outputs=outputs, name='gan-u-net-bn')  

    def train_step(self, data):

        x, y = data
        batch_size = tf.shape(x)[0]

        generated_images = self.generator(x, training = True)

        valid = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))
        valid += 0.05 * tf.random.uniform(tf.shape(valid))
        fake += 0.05 * tf.random.uniform(tf.shape(fake))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(y, training = True)
            d_loss_real = self.loss_fn(fake, predictions)
        grads = tape.gradient(d_loss_real, self.discriminator.trainable_weights)
        self.doptimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator(generated_images, training = True)
            d_loss_fake = self.loss_fn(valid, predictions)
        grads = tape.gradient(d_loss_fake, self.discriminator.trainable_weights)
        self.doptimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(x, training = True), training = True)
            g_loss = self.loss_fn(valid, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.goptimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {'d_loss':d_loss, 'g_loss':g_loss}

    def test_step(self, data):

        x, y = data
        batch_size = tf.shape(x)[0]

        generated_images = self.generator(x)

        valid = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))
        valid += 0.05 * tf.random.uniform(tf.shape(valid))
        fake += 0.05 * tf.random.uniform(tf.shape(fake))

        predictions = self.discriminator(y)
        d_loss_real = self.loss_fn(valid, predictions)
        predictions = self.discriminator(generated_images)
        d_loss_fake = self.loss_fn(fake, predictions)
        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

        predictions = self.discriminator(self.generator(x))
        g_loss = self.loss_fn(valid, predictions)
 
        return {'d_loss':d_loss, 'g_loss':g_loss}

def main():
    model = u_net()
    model.summary(print_fn=print)

if __name__ == '__main__':
    main()
