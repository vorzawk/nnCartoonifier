import cv2
import os
from cartoonify import cartoonify

"""
Takes all the images in the scratch folder and ouputs cartoon versions 
of them in the same folder
"""
folder = 'scratch'
real_inputs = []
cartoon_outputs = []
os.system('rm scratch/*_cartoon.jpg')
for filename in os.listdir(folder):
    img_rgb = cv2.imread(os.path.join(folder,filename))
    # If input file is img.jpg, name the output as img_cartoon.jpg
    split_filename = filename.split('.')
    outputFilename = split_filename[0] + '_cartoon.' + split_filename[1]

    if img_rgb is not None:
        output = cartoonify(img_rgb)
        real_inputs.append(img_rgb)
        cartoon_outputs.append(output)
        cv2.imwrite(os.path.join(folder,outputFilename), output)

import pickle
import numpy as np
# Use all images except the last 10 for training the model, use those for testing
with open('train_data', 'wb') as train_data:
    pickle.dump((np.array(real_inputs[:-10]), np.array(cartoon_outputs[:-10])), train_data)
with open('test_data', 'wb') as test_data:
    pickle.dump((np.array(real_inputs[-10:]), np.array(cartoon_outputs[-10:])), test_data)
