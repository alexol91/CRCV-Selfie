import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

from dataset_loader import get_img, img_loader
from common import *

from checkpoints import *
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
ctr = 0
dt = get_dataset(dataset_file_name)


'''
print("HEADER",list(dt.columns))
print("\n")
print("FIRST 10", dt[:10], sep="\n")
print("\n")
print("LAST 10", dt[-10:], sep="\n")
ip = next(img_loader(batch_size=1, io=2))
print(ip[1][0])
ip = next(img_loader(batch_size=1, io=2))
print(ip[1][0])
ip = next(img_loader(batch_size=1, io=2))
print(ip[1][0])
exit()
'''

# swtich window
io=2
model = load_model('weights/weights.96.h5')
#try_model(model, num_samples=6, io=io)

ip = []
num_samples = 7
for i in range(1, num_samples+1):
  img_path = 'samples/predict' + str(i) + '.jpg'
  ip.append(get_img(img_path))
ip = np.asarray(ip)
op = model.predict(ip)
print(list(dt.columns))
print(op)
decode_output(op, num_samples, io=io)

exit()