import os
import glob
# import zipfile
# import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras import Model
# from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import f1_score

import pandas as pd
from sklearn.model_selection import train_test_split

image_dir = 'ukraine-ml-bootcamp-2023/images/train_images/'
test_dir = 'ukraine-ml-bootcamp-2023/images/test_images/'
csv_path = 'ukraine-ml-bootcamp-2023/train.csv'

img_height=400
img_width=400

classes_count=6
        
class F1ScoreCallback(tf.keras.callbacks.Callback):
  def __init__(self, validation_generator):
    super().__init__()
    self.validation_generator = validation_generator
    self.f1_scores = []

  def on_epoch_end(self, epoch, logs=None):
    y_true = self.validation_generator.classes
    y_pred = self.model.predict(self.validation_generator)
    y_pred = tf.math.argmax(y_pred, axis=1).numpy()
    f1 = f1_score(y_true, y_pred, average='micro')
    self.f1_scores.append(f1)
    print(f'F1 Score: {f1}')

"""
Prepare generators
"""
def train_validation_generators():
  df = pd.read_csv(csv_path, dtype=str)
  train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

  train_datagen = ImageDataGenerator(
      rescale=1.0 / 255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
  )

  val_datagen = ImageDataGenerator(rescale=1.0 / 255)

  batch_size = 32

  train_generator = train_datagen.flow_from_dataframe(
      dataframe=train_df,
      directory=image_dir,
      x_col='image_id',
      y_col='class_6',
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',
      shuffle=True
  )

  val_generator = val_datagen.flow_from_dataframe(
      dataframe=val_df,
      directory=image_dir,
      x_col='image_id',
      y_col='class_6',
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',
      shuffle=False
  )

  return train_generator, val_generator

"""
Use pretrained Xception model trained on image.net
"""
def create_pre_trained_model():
  base_model = Xception(
    weights='imagenet', 
    include_top=False,
    input_shape = (img_height, img_width, 3)
  )

  # Freeze base model layers
  for layer in base_model.layers:
    layer.trainable = False

  return base_model

"""
Use final model
"""
def create_final_model(pre_trained_model):
  final_model = pre_trained_model.output
  final_model = tf.keras.layers.GlobalAveragePooling2D()(final_model)
  final_model = tf.keras.layers.Dense(128, activation='relu')(final_model)
  final_model = tf.keras.layers.Dropout(0.2)(final_model)
  output_layer = tf.keras.layers.Dense(classes_count, activation='softmax')(final_model)

  model = Model(
    inputs=pre_trained_model.input, 
    outputs=output_layer
  )

  # f1_metric = tf.keras.metrics.F1Score(
  #     threshold=0.5,
  #     average='micro'
  # )

  model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    #metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    metrics=['accuracy'] #, f1_metric]
  )
  
  return model

"""
Predicts classes for test images
"""
def predict_classes(model):
  test_dir_paths = glob.glob(os.path.join(test_dir, '*'))

  filenames = []
  predicted_classes = []

  for image_path in test_dir_paths:
      img = load_img(image_path, target_size=(img_height, img_width))
      img = img_to_array(img)
      img = preprocess_input(img)
      img = tf.expand_dims(img, axis=0)

      predictions = model.predict(img)
      predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

      filename = os.path.basename(image_path)

      filenames.append(filename)
      predicted_classes.append(predicted_class)

  df = pd.DataFrame({'image_id': filenames, 'class_6': predicted_classes})
  df.to_csv('test_submission.csv', sep=',', index=False)

"""
Main:
"""
train_generator, validation_generator = train_validation_generators()

model = create_final_model(
  pre_trained_model=create_pre_trained_model()
)

# f1_callback = F1ScoreCallback(validation_generator=validation_generator)

history = model.fit(
  train_generator,
  validation_data = validation_generator,
  epochs = 20,
  verbose = 1,
  # callbacks=[f1_callback]
)

print('predicting...')
predict_classes(model)

print('saving...')
model.save('yoga-prediction.keras')

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()
# plt.show()