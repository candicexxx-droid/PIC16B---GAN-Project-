import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from gan import GAN


# code that helps prevent my kernal from dying while training on gpu
# comment out if not needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#This file calculates the FID scores between the original images and different types of noises
#it saves the FID scores changes as a .png files 
####All images are converted to (0,255) range for RGB channels 

SAMPLE_SIZE = 5000
gan_model = GAN()
gan_model.dataset = gan_model.load_data()
training_cats = gan_model.real_samples(SAMPLE_SIZE)

# print(training_cats.shape)

from noise_function import Gaussian_noise, Gaussian_Blur, rect, apply_swirl
training_cats_255 = (training_cats*127.5 + 127.5).astype(np.uint8)
# plt.imshow(training_cats_255[0,:,:,:])

# training_cats_255_gnoise = Gaussian_noise(training_cats_255, 0.3)
# training_cats_gnoise = (training_cats_255_gnoise-127.5)/127.5
# plt.imshow(training_cats_gnoise[0,:,:,:])
# plt.imshow((training_cats[0,:,:,:]*127.5 + 127.5).astype(int))

from fid_noise import calculate_fid
model = gan_model.inception_classifier

#####FID for Gaussian Noise
alpha_gnoise = [0, 0.1, 0.25, 0.5]
fid_gnoise = []
for i in alpha_gnoise: 
    training_cats_255_gnoise = Gaussian_noise(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_gnoise)
    fid_gnoise.append(fid)

fig = sns.lineplot(x=alpha_gnoise, y=fid_gnoise)
fig.set(xlabel='alpha',
        ylabel='FID Score',
        title='FID Score with Variation in Gausian Noise')
plt.savefig('fid_GaussianNoise.png')
plt.clf()

####FID for Gaussian Blur 
alpha_gblur =  [0, 1, 3, 5]
fid_gblur = []
for i in alpha_gblur: 
    print('gblur: ', i)
    training_cats_255_gblur = Gaussian_Blur(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_gblur)
    fid_gblur.append(fid)

fig = sns.lineplot(x=alpha_gblur, y=fid_gblur)
fig.set(xlabel='alpha',
        ylabel='FID Score',
        title='FID Score with Variation in Gaussian Blur')
plt.savefig('fid_GaussianBlur.png')
plt.clf()

###FID for Random Rectangles
alpha_rect = [0.0, 0.25, 0.5, 0.75] 
fid_rect = []
for i in alpha_rect: 
    training_cats_255_rect = rect(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_rect)
    fid_rect.append(fid)
fig = sns.lineplot(x=alpha_rect, y=fid_rect)
fig.set(xlabel='alpha',
        ylabel='FID Score',
        title='FID Score with Variation in Random Rectangles')
plt.savefig('fid_Rect.png')
plt.clf()

### FID for Applying Swirl 
alpha_swirl = [0, 1, 2, 4]
fid_swirl= []
for i in alpha_swirl: 
    training_cats_255_swirl = apply_swirl(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_swirl)
    fid_swirl.append(fid)
fig = sns.lineplot(x=alpha_swirl, y=fid_swirl)
fig.set(xlabel='alpha',
        ylabel='FID Score',
        title='FID Score with Variation in Swirl')
plt.savefig('fid_Swirl.png')
plt.clf()
