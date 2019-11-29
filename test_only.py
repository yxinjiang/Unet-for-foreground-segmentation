from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.feature_extraction.image import extract_patches_2d

img_paths = list(paths.list_images('C:\\Users\\yixin\\Pictures\\QQplayerPic'))
img_path = img_paths[0]
img = Image.open(img_path)
#fig = plt.figure(figsize=(20,))
plt.imshow(img)
plt.show()
width,height = img.size

#filters = ['','Image.BILINEAR','Image.LANCZOS','Image.BICUBIC']
fig = plt.figure(figsize=(20,15))
for i in range(4):
    resize_img = img.resize((224,224),i)
    plt.subplot(2,2,i+1)
    plt.imshow(resize_img)
plt.show()

mask_path = 'D:\\GIT\\00Dataset\\Segmentation\\camvid\\LabeledApproved_full\\0001TP_006750_L.png'
mask = Image.open(mask_path)
plt.imshow(mask)
plt.show()

txt_path = ''
