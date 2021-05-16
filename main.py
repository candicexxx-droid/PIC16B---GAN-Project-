import tensorflow as tf
from gan import GAN

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

# TODO: try different hyperparamters
# hyper parameters
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
EPOCHS = 250
ALPHA = 2e-1
LABEL = class_names.index('deer')

# preventing my kernel from dying
# comment out if training on CPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load and train GAN model
# gan.load_model() can be called to use a previously trained model
gan_model = GAN(BATCH_SIZE, LEARNING_RATE, LABEL, ALPHA, EPOCHS)
gan_model.train()
gan_model.save_model()
gan_model.plot_generated_images('generated_images')
