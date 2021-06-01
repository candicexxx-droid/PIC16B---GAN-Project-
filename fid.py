# |%%--%%|
import tensorflow as tf
from gan import GAN


# code that helps prevent my kernal from dying while training on gpu
# comment out if not needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load last trained model to skip training time
gan_model = GAN()
gan_model.load_model()
# TODO: write code for interpolation and other
# FID related functions
# get fid score
gan_model.FID()
