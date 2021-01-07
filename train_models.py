from common import *
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from keras.models import load_model, Model, Sequential
from keras.layers import Flatten, Activation, Dropout, Dense, BatchNormalization, Input, Conv2D, MaxPool2D
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import pydot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from matplotlib import pyplot as plt
from keras_models import get_model, build_model
from dataset_loader import img_loader

io = 2

tensorboard_callback = TensorBoard(log_dir="logs",
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=False)
save_model_callback = ModelCheckpoint(os.path.join("weights", 'weights.{epoch:02d}.h5'),
                                      verbose=3,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='auto',
                                      period=8)

early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.001,
                                        patience=4,
                                        verbose=0, mode='auto')


BATCH_SIZE = 1024
STEPS_PER_EPOCH = 128
EPOCHS = 512
LAST_EPOCH = 64

if LAST_EPOCH:
  model = load_model('weights/weights.'+str(LAST_EPOCH)+'.h5')
else:
  model = build_model()
print(model)

'''
ip = next(img_loader(batch_size=1, io=2))
op = model.predict(ip[0])
'''
print('-- TRAINING --')
history = model.fit(
    img_loader(BATCH_SIZE, io=io),
    #     data_generator.batch_generator('train', batch_size=BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    initial_epoch=LAST_EPOCH,
    epochs=EPOCHS,
    validation_data=img_loader(BATCH_SIZE, io=io, mode="test"),
    validation_steps=STEPS_PER_EPOCH,
    callbacks=[save_model_callback, tensorboard_callback, early_stopping_callback],
    verbose=2
    #     workers=4,
    #     pickle_safe=True,
)
print('-- TRAINED --')
with open("history/pickled_history." + str(io) + ".pkl", "wb") as f:
    pickle.dump(history, f)

print('-- LOSS --')
print(history.history['loss'])
print(history.history['val_loss'])
'''
# Plot training & validation accuracy values
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model mean squared error')
plt.ylabel('Mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("train_metrics." + str(io) + ".png")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("test_metrics." + str(io) + ".png")
'''