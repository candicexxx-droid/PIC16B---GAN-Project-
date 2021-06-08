import matplotlib.pyplot as plt
import numpy as np
from gan import GAN

####All images are converted to (0,255) range for RGB channels 

training_cats = GAN.load_data()
print(training_cats.shape)

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
plt.ylabel('fid')
plt.savefig('fid_GaussianNoise')

####FID for Gaussian Blur 
alpha_gblur =  [0, 1, 2, 4]
fid_gblur = []
for i in alpha_gnoise: 
    training_cats_255_gblur = Gaussian_Blur(training_cats_255, i)
    fid = calculate_fid(model, training_cats_255,training_cats_255_gblur)
    fid_gblur.append(fid)
plt.plot(alpha_gblur, fid_gblur)
plt.xlabel('alpha')
plt.ylabel('fid')
plt.savefig('fid_GaussianBlur')

###FID for Random Rectangles
exam_num = training_cats_255.shape[0]
for j in range(exam_num): 
    
