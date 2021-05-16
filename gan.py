import numpy as np
from keras import datasets, optimizers, models, layers, Sequential
import matplotlib.pyplot as plt


# TODO: binary cross-entropy
class GAN:
    def __init__(self, batch_size, lr, label, alpha, epochs):
        self.batch_size = batch_size
        self.lr = lr
        self.label = label
        self.alpha = alpha
        self.epochs = epochs
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.gan = self.gan_model(self.generator, self.discriminator)
        self.dataset = self.load_data()
        self.half_batch = int(batch_size / 2)
        self.plot_generator_input = self.generator_input(16)

    def load_data(self):
        """
        Load the CIFAR-10 dataset with values in range [-1, 1]
        :param label: integer corresponding to class label
        """
        # loading a dataset of just one category to train a GAN model
        (train_images, train_labels), (_, _) = datasets.cifar10.load_data()
        train_images = train_images[train_labels.flatten() == self.label]
        train_images = (train_images.astype('float32') - 127.5) / 127.5

        return train_images

    def generator_model(self):
        """
        Create a generator model
        :return model: generator model
        """
        # TODO: use buffere value for 256
        # TODO: add/remove batch normalization layers
        model = Sequential([
            # layer 1 - 4x4 array
            layers.Dense(256 * 4 * 4, input_shape=(100, )),
            layers.Reshape((4, 4, 256)),
            layers.LeakyReLU(alpha=self.alpha),
            #             layers.BatchNormalization(),
            # layer 2 - 8x8 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            #             layers.BatchNormalization(),
            # layer 3 - 16x16 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            #             layers.BatchNormalization(),
            # layer 4 - 32x32 array
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                   padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            #             layers.BatchNormalization(),
            # layer 5 - output layer - 32x32x3 array
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
            layers.Conv2D(128, (3, 3), padding='same',
                          input_shape=(32, 32, 3)),
            layers.LeakyReLU(alpha=self.alpha),
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # layer 2
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # layer 3
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # layer 4
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=self.alpha),
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # output layer
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def gan_model(self, generator, discriminator):
        """
        Combine generator and discriminator to create a GAN model
        """
        disc_adam = optimizers.Adam(lr=self.lr, beta_1=0.5)
        discriminator.compile(loss='binary_crossentropy', optimizer=disc_adam)
        discriminator.trainable = False
        model = Sequential([generator, discriminator])
        gan_adam = optimizers.Adam(lr=self.lr, beta_1=0.5)
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
            if i % 5 == 0:
                self.plot_generated_images('epochs/Epoch_{}'.format(i))
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
        self.gan.summary()
