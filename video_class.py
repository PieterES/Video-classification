import tqdm
import random
import pathlib
import itertools
import collections
from load_data import list_files_from_zip_url, get_class, get_files_per_class, download_from_zip, split_class_lists, download_ucf_101_subset, format_frames, frames_from_video_file, FrameGenerator
import cv2
import einops # Provides a set of operations for manipulating tensors, reshaping, peruming, reducing dimensions
import numpy as np
import remotezip as rz
import seaborn as sns # Data visualization for informative statistical graphs
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf
import keras
from keras import layers
current_path = Path.cwd()
model_file_path = current_path / "path" / "to" / "saved" / "model"
class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension.

      Convolution is applied independently over the spatial dimensions of the input tensor, then independently over the temporal dimenison
      This allows the model to capture both spatial and temporal information separately, beneifical for tasks involving video analysis
      Where both spatial features and temporal dynamics are important

      filters: The number of output filters in the convolutional layers
      kernel_size: The size of the convolutional kernels, specified as a tuple (temporal, spatial)
      padding: The padding strategy for the convolutional layers
    """
    super().__init__()
    self.seq = keras.Sequential([ # Defines the sequential model containing two conv layers
        # Spatial decomposition
        layers.Conv3D(filters=filters, # Applies convolutional over the spatial dimensions of the input, number of output filters
                      kernel_size=(1, kernel_size[1], kernel_size[2]), # Sets the kernel size fo the spatial convolution
                      padding=padding), # Padding for spatial conv
        # Temporal decomposition
        layers.Conv3D(filters=filters, # Applies conv over temporal dimension
                      kernel_size=(kernel_size[0], 1, 1), # sets the kernel size for temporal convolution
                      padding=padding)
        ])

  def call(self, x): # The input tensor 'x' to be processed by the conv layer
      # the output tensor obtained by passing x through the sequential model self.seq
    return self.seq(x)

class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.

    A ResNet model is made from a sequence of residual blocks.
    The main branch performs the calculation, but is difficult for gradients to flow through.
    The residual branch bypasses the main calculation and mostly just adds the input to the output of the main branch.
    This avoids the vanishing gradient problem.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(), # Normalizes the activations of the previous layer
        layers.ReLU(), # Acitivation funciton that introduces non-linearity by setting negative to zero
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
  """
  def __init__(self, units):
      # units: the dimensionality of the output space (number of output units/neurons for the dense layer)
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  # Create an instance of Residual Main layer, then applies it to the input tensor
  out = ResidualMain(filters, kernel_size)(input)

  # Copies the input tensor, used later for addition
  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  # If the number of channels (last dimension) in the output tensor from the ResidualMain layer does not match the channels of the input, a projection is needed
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)
  # Adds the adjusted input tensor and the output tensor from the ResidualMain layer using the add function from the layers module
  # Implements the residual connection by adding the origianl input to the output of the residual block
  return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)
    # Initializes the 'Resizing' layer form keras layers module, used to perform the resizing operation on the video frames
  def call(self, video):
    """
      Use the einops library to resize the tensor.

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height,
    # w stands for width, and c stands for the number of channels.
    # einops library to parse the shape of the input video tensor

    # Parses the shape of the input video tensor and stores it in the old shape variable
    old_shape = einops.parse_shape(video, 'b t h w c')

    # reaaranges the dimensions of the input video tensor using rearrange function
    # flattens the batch and time dimensions into a single dimenison, so we get 4D instead of 5D
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    # resizes each frame in the batch to the specific height and width
    images = self.resizing_layer(images)
    # rearranges the dimensions back to the original shape
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        # ensures that the reshaped tensor retains the origianl number of frames
        t = old_shape['t'])
    return videos

HEIGHT = 224
WIDTH = 224

# Define the shape of the input tensor
input_shape = (None, 10, HEIGHT, WIDTH, 3)
# Create an input layer with specified shape
input = layers.Input(shape=(input_shape[1:]))

# x will be used to build the model by sequentially adding layers
x = input
# Adds a Conv2Plus1D layer to the model
x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
# Batch normalization helps stabalize and speed up training by normalizing the input to each layer
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
# Rezises the video frames to half
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Adds residual block which consists of two convolutional layers followed by normalization and activation layers
# helps learn complex features while mitigating the vanishing gradient problem.
# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

# Applies global average pooling across the spatial dimenisons and flattens the tensor to preprare for the final dense layer
x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)

# Adds a dense layer with 10 units, representing the output classes
x = layers.Dense(10)(x)

# Creates the keras model using the input layer and output tensor x
model = keras.Model(input, x)


URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ucf_101_subset(URL,
                        num_classes = 10,
                        splits = {"train": 30, "val": 10, "test": 10},
                        download_dir = download_dir)

n_frames = 10
batch_size = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], n_frames, training=True),
                                          output_signature = output_signature)


# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], n_frames),
                                        output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], n_frames),
                                         output_signature = output_signature)

test_ds = test_ds.batch(batch_size)

# Builds the Keras model by passing a batch of frames from the training dataset to determine the shape of the input tensors
frames, label = next(iter(train_ds))
model.build(frames)

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = ['accuracy'])
history = model.fit(x = train_ds,
                    epochs = 50,
                    validation_data = val_ds)



def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation'])

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()

plot_history(history)

model.evaluate(test_ds, return_dict=True)

def get_actual_predicted_labels(dataset):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type):
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize':(12, 12)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()

fg = FrameGenerator(subset_paths['train'], n_frames, training=True)
labels = list(fg.class_ids_for_name.keys())

actual, predicted = get_actual_predicted_labels(train_ds)
plot_confusion_matrix(actual, predicted, labels, 'training')

actual, predicted = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, labels, 'test')

def calculate_classification_metrics(y_actual, y_pred, labels):
  """
    Calculate the precision and recall of a classification model using the ground truth and
    predicted values.

    Args:
      y_actual: Ground truth labels.
      y_pred: Predicted labels.
      labels: List of classification labels.

    Return:
      Precision and recall measures.
  """
  cm = tf.math.confusion_matrix(y_actual, y_pred)
  tp = np.diag(cm) # Diagonal represents true positives
  precision = dict()
  recall = dict()
  for i in range(len(labels)):
    col = cm[:, i]
    fp = np.sum(col) - tp[i] # Sum of column minus true positive is false negative

    row = cm[i, :]
    fn = np.sum(row) - tp[i] # Sum of row minus true positive, is false negative

    if tp[i] + fp == 0:
        precision[labels[i]] = 0  # Handle the case where the denominator is zero
    else:
        precision[labels[i]] = tp[i] / (tp[i] + fp)

    if tp[i] + fn == 0:
        recall[labels[i]] = 0
    else:
        recall[labels[i]] = tp[i] / (tp[i] + fn) # Recall

  return precision, recall

precision, recall = calculate_classification_metrics(actual, predicted, labels) # Test dataset
print(precision)
print(recall)
