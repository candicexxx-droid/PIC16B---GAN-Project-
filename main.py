import tensorflow as tf
from gan import GAN

# hyper parameters
BATCH_SIZE = 128
LR1 = 5e-5
LR2 = 2e-4
EPOCHS = 500
ALPHA = 5e-1

# preventing my kernel from dying
# comment out if training on CPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load and train GAN model
# gan.load_model() can be called to use a previously trained model
gan_model = GAN(BATCH_SIZE, LR1, LR2, ALPHA, EPOCHS)
gan_model.train()
gan_model.save_model()
gan_model.plot_generated_images('generated_images')
gan_model.summary()
