"""
This file demonstrates how to use the GAN class to load a previously
trained model. To see an example of how to trian the model see training.py
"""
import tensorflow as tf
from gan import GAN


# code that helps prevent my kernal from dying while training on gpu
# comment out if not needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load last trained model to skip training time
gan_model = GAN()
gan_model.load_model()
# Save generated_images.png to directory containing a 4x4 grid of images
# generated by the model.
gan_model.plot_generated_images('generated_images')
# Prints the generator, discriminator, and GAN model summary
gan_model.summary()
