import tensorflow as tf
import imageio
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# from skimage.filters import gaussian
from scipy.linalg import sqrtm
import random
from skimage.transform import swirl

# im = imageio.imread('/Users/candicecai/Desktop/Sophomore_Spring_PIC_16B/PIC16B---GAN-Project-/cats/1.jpg')
# im_np = np.array(im)
# plt.imshow(im_np), plt.axis('off'), plt.show()

#####Gaussian_noise#####

def Gaussian_noise (image, alpha = 0.1): 
    """
    Add Gaussian noise to images

    Input Arguments: 
    image: one image (RGB channel range (0,255)) 
    alpha: a parameter for adding Gaussian noise 
        choices of alpha: 0, 0.1, 0.25, 0.3, 0.4 
    """
    image = (image - 127.5)/127.5
    mean = 0
    sigma = 1
    shape = image.shape
    gaussian = np.random.normal(mean, sigma, shape)  #generate gaussian noise with the same shape as the input images 

    interpolated = (1-alpha)*image + alpha*gaussian

    interpolated = (interpolated*127.5 + 127.5).astype(int)
    return interpolated



#####Gaussian_Blur#####

def Gaussian_Blur (images, ksize = 5, alpha = 0): 
  """
  Convolve images with a Gaussian kernel 

  Input Arguments: 
  image: all training data 
  ksize: kernel size 
  alpha: standard deviation of the Gaussian kernel 
    choices of alpha: 0, 1, 2, 4

  """
  
  # kernel = np.random.normal(mean, sigma, (ksize,ksize)) #Gaussian Kernel
  examp_num = images.shape[0]
  new_images = []
  for i in range(examp_num): 
    image = images[i].copy()
    new = cv.GaussianBlur(image,ksize = (ksize, ksize), sigmaX = alpha)
    new_images.append(new)
  new_images = np.array(new_images)
  return new_images





#####Add Random Rectangles#####



def rect(img, n_rect, share, hi=64, wi=64, chan=3, val=0.0):
    '''
    Apply n_rect numbers of black rectangles to images
    
    Input Arguments:
    img: training data(RGD channel range(0,225))
    n_rect: numbers of black rectangles want to add
    share: control the size of implanted rectangles(0-1)
    hi,wi,chan: shape of images
    '''



    def drop_rect(img_in, hi=64, wi=64, chan=3, share=0.5, positioning="random", val=0.0):
        '''
        Implant black rectangles to images
        
        Input Arguments:
        img_in: training data(RGD channel range(0,225))
        share: control the size of implanted rectangles(0-1)
        hi,wi,chan: shape of images
        '''
        img = img_in.copy()
        rhi = np.int(hi*share)
        rwi = np.int(wi*share)
        xpos = random.randint(0, hi-rhi)
        ypos = random.randint(0, wi-rwi)
        xdim = xpos + rhi
        ydim = ypos + rwi
        img = img.reshape(hi,wi, chan)
        img[xpos:xdim,ypos:ydim,:] = np.ones((rhi, rwi, chan))*val
        return img 




    img=img.astype(np.float).flatten()
    transf_data = np.zeros_like(img)
    for i in range(64):
        img = img.reshape(hi,wi,chan)
        transf_data = drop_rect(img, hi, wi, chan, share=share, val=val).flatten()
        for j in range(1,n_rect):
            img = transf_data.reshape(hi,wi,chan)
            transf_data = drop_rect(img, hi, wi, chan, share=share, val=val).flatten()
    return transf_data.reshape(64,64,3).astype(int)

#####Swirl#####



# In[106]:
def apply_swirl(img, n_swirls, radius, strength, hi=64, wi=64, chan=3):
    '''
    Apply Swirl to images
    
    Input Arguments:
    img: training data(RGD channel range(0,225))
    n_swirls: number of swirls applied
    radius: control the size of swirls(0-hi)
    hi,wi,chan: shape of images
    '''
    def lokal_swirl(img_in, n_swirls, radius, strength, hi=64, wi=64, chan=3, corr_size=3):
        '''
        Swirl the images
        
        Input Arguments:
        img_in: training data(RGD channel range(0,225))
        n_swirls: number of swirls applied
        radius: control the size of swirls(0-hi)
        hi,wi,chan: shape of images
        '''
        img = img_in.copy()
        size = corr_size
        for i in range(n_swirls):
            sign = np.sign(np.random.rand(1) - 0.5)[0]

            xpos = hi // 2
            ypos = wi // 2
            center = (xpos,ypos)
            img = swirl(img, rotation=0, strength=sign*strength, radius=radius, center=center)
            img[0:size] = img_in[0:size]
            img[-(size+1):] = img_in[-(size+1):]
            img[:,0:size] = img_in[:,0:size]
            img[:,-(size+1):] = img_in[:,-(size+1):]
        return img
    img=img.astype(np.float).flatten()
    transf_data = np.zeros_like(img)
    for i in range(64):
        img_in = img.reshape(hi,wi,chan)
        img = lokal_swirl(img_in, n_swirls, radius, strength, hi=64, wi=64, chan=4)
        transf_data = img.flatten()
    return transf_data.reshape(64,64,3).astype(int)

#####Test#######
# plt.imshow(Gaussian_noise(im_np)), plt.axis('off'), plt.show()
# plt.imshow(Gaussian_Blur(im_np)), plt.axis('off'), plt.show()
# plt.imshow(rect(im_np,n_rect=2, share=0.10)), plt.axis('off'), plt.show()
# plt.imshow(apply_swirl(im_np,n_swirls=1,radius=70,strength=4.0)), plt.axis('off'), plt.show()





