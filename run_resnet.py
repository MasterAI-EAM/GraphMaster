import os, shutil, csv
import numpy as np
import pandas as pd
import cv2
import optuna

import tensorflow as tf
from keras.models import Model, Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models
from keras.layers import Flatten, Dense, Conv2D, Input, ZeroPadding2D, AveragePooling2D, BatchNormalization, Activation, add


def objective(trial):
  # /srv/scratch/z5293104/graphMaster_data
  testdf = pd.read_csv('/home/weijian-volume/weijian/weijiandl/git/GraphMaster/data/annotation/train.txt')

  testdf = testdf.to_numpy()

  TEST_DIR = '/home/weijian-volume/weijian/weijiandl/git/GraphMaster/data/images/'

  # OUTPUT_DIR = '../DocFigure_dataset/GraphPlots'

  # if not os.path.exists(OUTPUT_DIR):
  #     os.mkdir(OUTPUT_DIR)
      
  # for file in graphplotlist:
  #     shutil.copy(file, OUTPUT_DIR)

  # prepare training
  train_imgs = []
  train_labels = []

  for i in range(len(testdf)):
      
      if os.path.exists(TEST_DIR + testdf[i][0]):
          img = cv2.imread(TEST_DIR + testdf[i][0], cv2.IMREAD_COLOR)
  #         img = cv2.imread(TEST_DIR + testdf[i][0], cv2.IMREAD_GRAYSCALE)
          if img is not None :
  #             print(TEST_DIR + testdf[i][0])
              img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA).reshape(112, 112, 3)

          #     img = image.load_img(TEST_DIR + testdf[i][0], target_size=(224, 224))
          #     x = image.img_to_array(img)
          #     x = np.expand_dims(x, axis=0)
          #     x = preprocess_input(x)

              train_imgs.append(img)
              if testdf[i][1] == ' Graph plots':
                  train_labels.append([1, 0, 0, 0])
              elif testdf[i][1] == ' Tables':
                  train_labels.append([0, 1, 0, 0])
              elif testdf[i][1] == ' Bar plots':
                  train_labels.append([0, 0, 1, 0])
              elif testdf[i][1] == ' Algorithm':
                  train_labels.append([0, 0, 0, 1])
              else:
                  train_labels.append([0, 0, 0, 0])
  # train_imgs = np.array(train_imgs)
  # train_labels = np.array(train_labels)

  p = np.random.permutation(len(train_imgs))
  train_imgs = np.array(train_imgs)[p]
  train_labels = np.array(train_labels)[p]


  # define model
  def conv_bn(x, nb_filters, kernel_size, padding="same", strides=(1, 1), name=None):
      if name is not None:
          bn_name = name + "_bn"
          conv_name = name + "_conv"
      else:
          bn_name = None
          conv_name = None
      x = Conv2D(nb_filters, kernel_size, padding=padding, strides=strides, name=conv_name)(x)

      x = BatchNormalization(axis=-1, name=bn_name)(x)
      x = Activation('relu')(x)

      return x


  def bottle_block18(input, nb_filters, padding="same", strides=(1, 1), with_conv_shortcut=False):
      k1, k2 = nb_filters
      x = conv_bn(input, k1, (3, 3), padding=padding, strides=strides)
      x = conv_bn(x, k2, (3, 3), padding=padding)
      if with_conv_shortcut:
          shortcut = conv_bn(input, k2, (1, 1), padding=padding, strides=strides)
          x = add([x, shortcut])
      else:
          x = add([x, input])
      return x

  def resNet_customized(height, width, channels):
      input = Input(shape=(height, width, channels))

      x = input

      x = ZeroPadding2D((1, 1))(input)
      x = conv_bn(x, nb_filters=64, kernel_size=(3, 3), padding="valid", strides=(1, 1))

      x = bottle_block18(x, (64, 64), strides=(1, 1), with_conv_shortcut=True)
      x = bottle_block18(x, (64, 64), strides=(1, 1), with_conv_shortcut=True)
      x = bottle_block18(x, (64, 64), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (64, 64), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (64, 64), strides=(1, 1), with_conv_shortcut=True)

      x = bottle_block18(x, (128, 128), strides=(2, 2), with_conv_shortcut=True)
      x = bottle_block18(x, (128, 128), strides=(1, 1), with_conv_shortcut=True)
      x = bottle_block18(x, (128, 128), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (128, 128), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (128, 128), strides=(1, 1), with_conv_shortcut=True)

      x = bottle_block18(x, (256, 256), strides=(2, 2), with_conv_shortcut=True)
      x = bottle_block18(x, (256, 256), strides=(1, 1), with_conv_shortcut=True)
      x = bottle_block18(x, (256, 256), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (256, 256), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (256, 256), strides=(1, 1), with_conv_shortcut=True)

      x = bottle_block18(x, (512, 512), strides=(2, 2), with_conv_shortcut=True)
      x = bottle_block18(x, (512, 512), strides=(1, 1), with_conv_shortcut=True)
      x = bottle_block18(x, (512, 512), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (512, 512), strides=(1, 1), with_conv_shortcut=True)
      # x = bottle_block18(x, (512, 512), strides=(1, 1), with_conv_shortcut=True)

      x = AveragePooling2D((7, 7))(x)

      x = Flatten()(x)
      # x = Dense(128, activation="relu")(x)
      output = Dense(4, activation='sigmoid')(x)

      model = Model(inputs=input, outputs=output)

      return model

  model = resNet_customized(112, 112, 3)

  model.summary()

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 0, 1)),loss='categorical_crossentropy',metrics=['accuracy'])

  # history = model.fit(train_imgs, train_labels, validation_split=0.3, epochs=30)
  history = model.fit(train_imgs, train_labels, batch_size=trial.suggest_int("batch_size", 0, 70), epochs=30)
  model.save("../DocFigure_dataset/resNet_customized34")

  # testing
  testset = pd.read_csv('/home/weijian-volume/weijian/weijiandl/git/GraphMaster/data/annotation/test.txt')

  testset = testset.to_numpy()

  IMAGE_DIR = '/home/weijian-volume/weijian/weijiandl/git/GraphMaster/data/images/'


  test_imgs = []
  test_labels = []

  for i in range(len(testset)):
      
      if os.path.exists(IMAGE_DIR + testset[i][0]):
          img = cv2.imread(IMAGE_DIR + testset[i][0], cv2.IMREAD_COLOR)
  #         img = cv2.imread(IMAGE_DIR + testset[i][0], cv2.IMREAD_GRAYSCALE)
          if img is not None :
  #             print(IMAGE_DIR + testset[i][0])
              img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA).reshape(112, 112, 3)

          #     img = image.load_img(IMAGE_DIR + testset[i][0], target_size=(224, 224))
          #     x = image.img_to_array(img)
          #     x = np.expand_dims(x, axis=0)
          #     x = preprocess_input(x)

              test_imgs.append(img)
              if testset[i][1] == ' Graph plots':
                  test_labels.append([1, 0, 0, 0])
              elif testset[i][1] == ' Tables':
                  test_labels.append([0, 1, 0, 0])
              elif testset[i][1] == ' Bar plots':
                  test_labels.append([0, 0, 1, 0])
              elif testset[i][1] == ' Algorithm':
                  test_labels.append([0, 0, 0, 1])
              else:
                  test_labels.append([0, 0, 0, 0])

  test_imgs = np.array(test_imgs)
  test_labels = np.array(test_labels)
  loss, accuracy= model.evaluate(test_imgs, test_labels, verbose=False)
  print(accuracy)

  with open("acc.txt", 'a') as f:
    f.write(str(accuracy))
    f.write("\n")

  return accuracy

def main():
  search_space = {
      "lr": [1e-3, 1e-4, 1e-5],
      "batch_size": [16, 32, 64]
  }

  study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="maximize")
  study.optimize(objective)

  best = study.best_trial
  best_acc = best.value
  print(f"best accuracy: {best_acc}")
  print(f"best hparams: ")
  for key, value in best.params.items():
    print("    {}: {}".format(key, value))



if __name__ == "__main__":
  main()