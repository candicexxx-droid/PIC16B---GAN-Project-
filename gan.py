import os
import numpy as np
from keras import optimizers, models, layers, Sequential
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


class GAN:
    """
    Generative Adversarial Network class used to load a cat dataset and
    train a model.
    """
    def __init__(self, batch_size, lr1, lr2, alpha, epochs):
        self.batch_size = batch_size
        self.lr1 = lr1
        self.lr2 = lr2
        self.alpha = alpha
        self.epochs = epochs
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.gan = self.gan_model(self.generator, self.discriminator)
        self.dataset = self.load_data()
        self.half_batch = int(batch_size / 2)
        self.plot_generator_input = self.generator_input(16)

    @staticmethod
    def load_data():
        """
        Load cat dataset from folder with values in range [-1, 1]
        """

        if os.path.exists('data.npy'):
            images = np.load('data.npy')

        else:
            images = np.array([], dtype=np.uint8).reshape((0, 64, 64, 3))
            for file in os.scandir('cats'):
                if file.path.endswith('.jpg'):
                    images = np.append(images, load_img(file))
            images = images.reshape(15747, 64, 64, 3)
            images = (images.astype('float32') - 127.5) / 127.5

            np.save('data.npy', images)

        return images

    def generator_model(self):
        """
        Create a generator model
        """
        # tried to impliment batch normalization but didn't trail well
        model = Sequential([
            # layer 1 - 4x4 array
            layers.Dense(256 * 4 * 4, input_shape=(100, )),
            layers.Reshape((4, 4, 256)),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 2 - 8x8 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 3 - 16x16 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 4 - 32x32 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 5 - 64x64 array
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # output layer - 32x32x3 array
            layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
        ])

        return model

    def discriminator_model(self):
        """
        Create a discriminator model
        :return model: discriminator model
        """
        model = Sequential([
            # layer 1
            layers.Conv2D(64, (3, 3), padding='same', input_shape=(64, 64, 3)),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 2
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 3
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 4
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layer 5
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # output layer
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def gan_model(self, generator, discriminator):
        """
        Combine generator and discriminator to create a GAN model
        """
        disc_adam = optimizers.Adam(lr=self.lr1, beta_1=0.5)
        discriminator.compile(loss='binary_crossentropy', optimizer=disc_adam)
        discriminator.trainable = False
        model = Sequential([generator, discriminator])
        gan_adam = optimizers.Adam(lr=self.lr2, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=gan_adam)
        return model

    def real_samples(self):
        """
        Load real samples from the dataset
        """
        indexes = np.random.randint(self.dataset.shape[0],
                                    size=self.half_batch)
        real_images = self.dataset[indexes]
        return real_images

    @staticmethod
    def generator_input(sample_size):
        """
        Create input to be used by the generator
        """
        return np.random.randn(sample_size, 100)

    def train(self):
        """
        Train the GAN model
        """
        batches = int(self.dataset.shape[0] / self.batch_size)
        for i in range(self.epochs):
            print('Training Epoch: {}'.format(i))
            for _ in range(batches):
                # train on real samples
                real_images = self.real_samples()
                real_labels = np.ones(shape=(self.half_batch, 1))
                self.discriminator.train_on_batch(real_images, real_labels)
                # trian on fake samples
                fake_input = self.generator_input(self.half_batch)
                fake_images = self.generator.predict(fake_input)
                fake_labels = np.zeros(shape=(self.half_batch, 1))
                self.discriminator.train_on_batch(fake_images, fake_labels)
                # train GAN
                gan_input = self.generator_input(self.batch_size)
                gan_label = np.ones(shape=(self.batch_size, 1))
                self.gan.train_on_batch(gan_input, gan_label)
            if (i + 1) % 5 == 0:
                self.plot_generated_images('epochs/Epoch_{}'.format(i + 1))

    def save_model(self):
        """
        Save a GAN model for later use
        """
        self.gan.save('gan_model')

    def load_model(self):
        """
        Load a previously saved GAN model
        """
        self.gan = models.load_model('gan_model')

    def plot_generated_images(self, filename):
        """
        Save a png of generated images
        """
        # generator fake images
        images = self.generator.predict(self.plot_generator_input)
        images = (images + 1) / 2.0

        # plot generator images on subplots
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.axis('off')
            plt.imshow(images[i])

        # save figure as png
        plt.savefig(filename)

    def plot_individual_image(self):
        """
        Plot an individual image number 0-15 corresponding to images
        ploted by self.plot_generated_images()
        """
        ...

    def summary(self):
        """
        Print model summary
        """
        print("Generator Model Summary:")
        self.generator.summary()
        print('Discriminator Model Summary:')
        self.discriminator.summary()
        print('GAN Model Summary:')
        self.gan.summary()
