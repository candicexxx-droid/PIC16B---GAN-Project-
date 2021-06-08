import matplotlib.pyplot as plt
import numpy as np
from gan import GAN


#This file calculates the FID scores between the original images and different types of noises
#it saves the FID scores changes as a .png files 
####All images are converted to (0,255) range for RGB channels 


training_cats = GAN.load_data()
# print(training_cats.shape)

from noise_function import Gaussian_noise, Gaussian_Blur, rect, apply_swirl
training_cats_255 = (training_cats*127.5 + 127.5).astype(np.uint8)
plt.imshow(training_cats_255[0,:,:,:])

# training_cats_255_gnoise = Gaussian_noise(training_cats_255, 0.3)
# training_cats_gnoise = (training_cats_255_gnoise-127.5)/127.5
# plt.imshow(training_cats_gnoise[0,:,:,:])
# plt.imshow((training_cats[0,:,:,:]*127.5 + 127.5).astype(int))

from fid_noise import calculate_fid
gan = GAN()
model = gan.inception_classifier

#####FID for Gaussian Noise
alpha_gnoise = [0, 0.1, 0.25, 0.5]
fid_gnoise = []
for i in alpha_gnoise: 
    training_cats_255_gnoise = Gaussian_noise(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_gnoise)
    fid_gnoise.append(fid)
plt.plot(alpha_gnoise, fid_gnoise)
plt.xlabel('alpha')
plt.ylabel('FID Score')
plt.savefig('fid_GaussianNoise.png')

####FID for Gaussian Blur 
alpha_gblur =  [0, 1, 2, 4]
fid_gblur = []
for i in alpha_gnoise: 
    training_cats_255_gblur = Gaussian_Blur(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_gblur)
    fid_gblur.append(fid)
plt.plot(alpha_gblur, fid_gblur)
plt.xlabel('alpha')
plt.ylabel('FID Score')
plt.savefig('fid_GaussianBlur.png')

###FID for Random Rectangles
alpha_rect = [0.0, 0.25, 0.5, 0.75] 
fid_rect = []
for i in alpha_rect: 
    training_cats_255_rect = rect(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_rect)
    fid_rect.append(fid)
plt.plot(alpha_rect, fid_rect)
plt.xlabel('alpha')
plt.ylabel('FID Score')
plt.savefig('fid_rect.png')

### FID for Applying Swirl 
alpha_swirl = [0, 1, 2, 4]
fid_swirl= []
for i in alpha_swirl: 
    training_cats_255_swirl = apply_swirl(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_swirl)
    fid_swirl.append(fid)
plt.plot(alpha_swirl, fid_swirl)
plt.xlabel('alpha')
plt.ylabel('FID Score')
plt.savefig('fid_swirl.png')
