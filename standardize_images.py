import os
import glob
from time import time
import cv2

total_imgs = 0
processed_imgs = 0
start_time =time()
print('Site Data Set Preparation, Resizing to 800 x 600')
for a_class, a_dir in enumerate(['red', 'not_red', 'not_light']):
    print('Converting', a_dir, 'images', 'total images: ', total_imgs, 'resized images: ', processed_imgs)
    for a_filename in glob.glob('./site_dataset/{}/*.jpg'.format(a_dir)):
        a_filename = '/'.join(a_filename.split('\\'))
        img = cv2.imread(a_filename)
        if(img.shape[0] != 600) or (img.shape[1] != 800):
            print(a_filename, 'dimensions: ', img.shape[0], 'x', img.shape[1])
            img = cv2.resize(img, (800,600))
            cv2.imwrite(a_filename, img)
            processed_imgs += 1
        total_imgs += 1
            
print('Total time taken: ', time() - start_time, 'total images: ', total_imgs, 'resized images: ', processed_imgs)

total_imgs = 0
processed_imgs = 0
start_time =time()
print('Simulator Data Set Preparation, Resizing to 800 x 600')
for a_class, a_dir in enumerate(['green', 'not_light', 'red', 'yellow']):
    print('Converting', a_dir, 'images', 'total images: ', total_imgs, 'resized images: ', processed_imgs)
    for a_filename in glob.glob('./sim_dataset/{}/*.jpg'.format(a_dir)):
        a_filename = '/'.join(a_filename.split('\\'))
        img = cv2.imread(a_filename)
        if(img.shape[0] != 600) or (img.shape[1] != 800):
            print(a_filename, 'dimensions: ', img.shape[0], 'x', img.shape[1])
            img = cv2.resize(img, (800,600))
            cv2.imwrite(a_filename, img)
            processed_imgs += 1
        total_imgs += 1
            
print('Total time taken: ', time() - start_time, 'total images: ', total_imgs, 'resized images: ', processed_imgs)

