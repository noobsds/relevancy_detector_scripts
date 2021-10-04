import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

batch_size = 32
img_height = 360
img_width = 360

data_dir = "../image_test_set/images"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.15,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.15,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# train_datagen = ImageDataGenerator(
#     rescale=1./255, featurewise_center=False, 
#     featurewise_std_normalization=False, zoom_range=0.1, brightness_range=(0.8,1.2), width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=20, horizontal_flip=True, vertical_flip=True)


class_names = train_ds.class_names
print(class_names)


data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    tf.keras.layers.experimental.preprocessing.RandomFlip("vertical", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1)
  ]
)

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  data_augmentation,
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(
  optimizer='adam',
  loss="binary_crossentropy",
  metrics=['binary_accuracy'])


checkpoint_path = "training_1/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 save_best_only=False,
                                                 verbose=1, save_freq=5*batch_size)


model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100,
  callbacks=[cp_callback]
)


print("Model fitting done!")
print("Predicting on an irrelevant picture gives the following score (lower better):")
img = tf.keras.preprocessing.image.load_img(
    "../image_test_set/images/irrelevant_for_object_detection/apple_0_beating_heart.png", target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
print(img_array)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)

print("Predicting on a relevant picture gives the following score (higher better):")
img = tf.keras.preprocessing.image.load_img(
    "../image_test_set/images/relevant_for_object_detection/1RxzYsm.jpeg", target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)
