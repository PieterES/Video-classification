import tqdm # Progress bar for loops
import random
import pathlib # Simplifies file management tasks
import itertools # Tool for constructing and interacting with iterators for efficient looping
import collections # Offers specialized container datatypes such as Counter, deque

import os # Provies way to use the operating system-dependent functionality
import cv2 # OpenCV, open source computer vision and machien learning software (image and video processing)
import numpy as np
import remotezip as rz # Handle ZIP files on a remote server

import tensorflow as tf # Machine learning framework

# Some modules to display an animation using imageio.
import imageio # Library for reading and writing images, creating animations or processing video frames
from IPython import display # Embed images and videos within Jupyter environments
from urllib import request # Module for opening and reading URLs
from tensorflow_docs.vis import embed # embedding interaction visualizations in Jupyter notebooks

def list_files_from_zip_url(zip_url):
  """ List the files in each class of the dataset given a URL with the zip file.

    Args:
      zip_url: A URL from which the files can be extracted from.

    Returns:
      List of files in each of the classes.
  """
  files = [] # Empty list to store names of files
  with rz.RemoteZip(zip_url) as zip: # content manager ('with' statement) to open a remote zip file, creating an
      # instance of RemoteZip, allowing interaction with the zip file directly over the network within this block
    for zip_info in zip.infolist(): # iterates over each item in the list of zipinfo returned by zip.infolist() (metadata)
      files.append(zip_info.filename) # For each zipinfo object, its filename attribute is appended to the files list
  return files


def get_class(fname):
  """ Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Returns:
      Class that the file belongs to.

    Each file name looks like 'v_ApplyEyeMakeup_g01_c01.avi
    Accessing the third to last element gives the class name of the action cateogry
  """
  return fname.split('_')[-3]

def get_files_per_class(files):
  """ Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Returns:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list) # Initializes a defaultdict from the collections module with
  # list instances as the defaul value, useful for accumulating filesnames without needing to check if the key exists.
  for fname in files: # Iterates over each filename in the input list 'files'.
    class_name = get_class(fname) # get the class name of the file_name
    files_for_class[class_name].append(fname) # Add the current filename tot he list of files associated with its class
  return files_for_class

def select_subset_of_classes(files_for_class, classes, files_per_class):
  """ Create a dictionary with the class name and a subset of the files in that class.

    Args:
      files_for_class: Dictionary of class names (key) and files (values).
      classes: List of classes.
      files_per_class: Number of files per class of interest.

    Returns:
      Dictionary with class as key and list of specified number of video files in that class.
  """
  files_subset = dict() # Initializes empty dictionary, which will be populated wtih class names as keys and a list of files as valeus

  for class_name in classes: # For each iterations, class name is set to the current class form the classes list
    class_files = files_for_class[class_name] # Retrieves the list of all files belonging to the current class from the files for class dictionary
    files_subset[class_name] = class_files[:files_per_class] # slices the list of class files to only include the first files per class elements

  return files_subset

def download_from_zip(zip_url, to_dir, file_names):
  """ Download the contents of the zip file from the zip URL.

    Args:
      zip_url: A URL with a zip file containing data.
      to_dir: A directory to download data to.
      file_names: Names of files to download.
  """
  with rz.RemoteZip(zip_url) as zip: # Opens the ZIP file located at zip_url as zip in this block only
    for fn in tqdm.tqdm(file_names): # tierates over each filename, wrapping with tqdm to display a progress bar
      class_name = get_class(fn) # for each filename, extract the class name
      zip.extract(fn, str(to_dir / class_name)) # Extracts the current file from the ZIP archive to a directory path
      # constructed by joining 'to_dir', the target directory with the class name
      unzipped_file = to_dir / class_name / fn # Constructs the path for the extracted file, which will be used for renaming

      fn = pathlib.Path(fn).parts[-1] # Reassigns 'fn' to its base name, stripping any directory components
      output_file = to_dir / class_name / fn # Constrcuts the finale output file path
      unzipped_file.rename(output_file) # Renames the extracted file ot the final output file path

def split_class_lists(files_for_class, count):
  """ Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Returns:
      Files belonging to the subset of data and dictionary of the remainder of files that need to be downloaded.
  """
  split_files = [] # Empty list, holds the selected subset of files specified by count
  remainder = {} # Empty dict, will map each class to a list of files not included in the split_files subset
  for cls in files_for_class: # for each class in the dictionary
    split_files.extend(files_for_class[cls][:count]) # Extends the split_files list with the first 'count' files
    # from the current class, slices the list of files for the current class to include count elements
    remainder[cls] = files_for_class[cls][count:] # The list of files of the class cls beyond the first count, files not included in the previous subset
  return split_files, remainder

def download_ucf_101_subset(zip_url, num_classes, splits, download_dir):
  """ Download a subset of the UCF101 dataset and split them into various parts, such as
    training, validation, and test.

    Args:
      zip_url: A URL with a ZIP file with the data.
      num_classes: Number of labels.
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data
              (value is number of files per split).
      download_dir: Directory to download data to.

    Return:
      Mapping of the directories containing the subsections of data.
  """
  files = list_files_from_zip_url(zip_url) # Returns a list of all files names contained within the zip archive
  for f in files: # iterates over each file name in files
    path = os.path.normpath(f) # normalizes the files path, siplifying any redundant separators:
    # (dir//file -> dir/file and dir/sub/../file -> dir/file)
    tokens = path.split(os.sep) # Splits the normalized path into components using OS-specific path separator
    if len(tokens) <= 2: # If the number of tokens is two or lower, unvalid file
      files.remove(f) # Remove that item from the list if it does not meet criteria of being valid

  files_for_class = get_files_per_class(files) # organizes the filtered list by class. Each key is a class name, value is list of file names belonging to that class

  classes = list(files_for_class.keys())[:num_classes] # Extracts class names, converted to a list and selecting the first 'num_classes' classes

  for cls in classes: # for each class in classes
    random.shuffle(files_for_class[cls]) # Randomly shuffles the list of file names, removing inherent ordering

  # Only use the number of classes you want in the dictionary
  files_for_class = {x: files_for_class[x] for x in classes} # Comprehension that rebuilds the files for class dictionary
  # to only include the selected classes list

  dirs = {} # Empty dictionary to store paths to directories where files for each split will be downloaded
  for split_name, split_count in splits.items(): # Iterates over each split defined in the splits dictionary
    print(split_name, ":") # Visual cue
    split_dir = download_dir / split_name # Constructs the path for the current splits directionary
    split_files, files_for_class = split_class_lists(files_for_class, split_count) # Selects split_count number of files for
    # the current split and updates files_for_class to reflect the remaining files
    # Commented out so we don't download again
    # download_from_zip(zip_url, split_dir, split_files) # Downloads the selected files for the current split from the ZIP archive to the designated directory
    dirs[split_name] = split_dir # Stores the path to the directory for the current split in the dirs dictionary

  return dirs # Returns the dirs dictioanry, mapping split names ot their paths, easy reference

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32) # Converts the datatype frame to float32
  frame = tf.image.resize_with_pad(frame, *output_size) # Resizes the image to a target size (output_size) while maintaing its aspect ratio
  # if necessary, it adds padding  to reach the desired dimensions
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = [] # Empty list to store processed frames
  src = cv2.VideoCapture(str(video_path)) # Uses OpenCV's 'VideoCapture' to load the video from videopath

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT) # Retrieves the total frame count of the video using the
  # CAP_PROP_FRAME_COUNT property to ensure the fram extraction process does not exceed the video's length

  need_length = 1 + (n_frames - 1) * frame_step # Determines the total length needed to extract the specified number of frames
  # considering frame_step, helps deciding the starting point for frame extraction.

  if need_length > video_length:
    start = 0 # If the need_length is more than the video length, start at frame 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1) # Otherwise select a random starting point

  src.set(cv2.CAP_PROP_POS_FRAMES, start) # Positions the video reader to start extracting from the 'start' frame index
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read() # Reads the first frame, if succesful (ret = True)
  result.append(format_frames(frame, output_size)) # The frame is then processed using format_frames and added to result

  for _ in range(n_frames - 1): # For each subsequent frame to be extracted
    for _ in range(frame_step): # The loop skips frame_step frames
      ret, frame = src.read() # By reading for each frame in frame_step, the reading continues to the next frame
    if ret: # But only processes if after skipping frame_step amount of frames and if the read was succesfull
      frame = format_frames(frame, output_size) # Formats frame
      result.append(frame) # Adds to result
    else: # If reading was unsuccesfull, like end of video
      result.append(np.zeros_like(result[0])) # Add same size but all 0's
  src.release() # Releases resources associated with the video file, closing the file
  result = np.array(result)[..., [2, 1, 0]] # Converst the list of frames to a NumPy array and reorders the color channels from OpenCV's BGR to RGB

  return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path # Stores the path to directory containing the files
    self.n_frames = n_frames # Stores the number of frames to extract form each video
    self.training = training # Boolean flag to indicate if the generator is in training mode, affecting data shuffling
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir())) # Extracts and sorts the names of the directories
    # within 'path' by iterating through items in 'path', selecting directories and extracting their names
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names)) # Maps class names to numerical IDs for use as labels

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi')) # Compiles a list of video file paths
    classes = [p.parent.name for p in video_paths] # Extracts class labels based on the directory name of each video file
    return video_paths, classes # Returns a pair of lists, video file paths and their class names

  def __call__(self): # Makes the framegenerator instance callable, allowing it to function like a generator
    video_paths, classes = self.get_files_and_class_names() # Retrives video paths and class names

    pairs = list(zip(video_paths, classes)) #  Zip takes one or more iterables as inputs and returns an iterator of tuples
    # Each tuple contains the elements from the inputs iterables that are at the same position, pairing elements into tuples
    # Then converting to a list of tuples, making it easier to work with
    if self.training: # If self.training is True, shuffle the pairs, randomizing the order for training purposes
      random.shuffle(pairs)

    for path, name in pairs: # For each path and name in pairs
      video_frames = frames_from_video_file(path, self.n_frames) # Extract the frames we want for training
      label = self.class_ids_for_name[name] # Encode labels, corresponding numerical class ID based on name
      yield video_frames, label # 'yeild' statement is used to produce a sequence of values lazily, one at a time and on demand
      # more memory-efficient than generating and storing the entire sequence at once
def main():
  pass
if __name__ == "__main__":
  URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip' # zip file containing UCF101 videos
  """"
  Creates a list of all files in the ZIP file located in the URL
  List comprehension to only select those that end with .avi
  List slicing to selec thte first 10 elements as example.
  """
  files = list_files_from_zip_url(URL)
  files = [f for f in files if f.endswith('.avi')]
  files[:10]

  NUM_CLASSES = 10
  FILES_PER_CLASS = 50


  files_for_class = get_files_per_class(files) # organizes all the relevant video files by their class
  classes = list(files_for_class.keys()) # list of class names of the videos

  files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS)
  list(files_subset.keys())

  # Downloads a subset of the UCF 101 videos, commented out
  download_dir = pathlib.Path('./UCF101_subset/')
  subset_paths = download_ucf_101_subset(URL,
                                         num_classes = NUM_CLASSES,
                                         splits = {"train": 30, "val": 10, "test": 10},
                                         download_dir = download_dir)

  video_count_train = len(list(download_dir.glob('train/*/*.avi')))
  video_count_val = len(list(download_dir.glob('val/*/*.avi')))
  video_count_test = len(list(download_dir.glob('test/*/*.avi')))
  video_total = video_count_train + video_count_val + video_count_test
  print(f"Total videos: {video_total}")

  # Creates instance of the FrameGenerator class to generate frames from videos located in 'train'
  fg = FrameGenerator(subset_paths['train'], 10, training=True)

  # Calling fg() generator, implementing the call method, starting the process of generating frames and labels
  # next() fetches the first output from the generator
  frames, label = next(fg())

  print(f"Shape: {frames.shape}")
  print(f"Label: {label}")

  # Create the training set

  """
  # Defines the expected structure and data types of the items yielded by teh generator, specifies that frames yielded
  # by the generator will be 40dimensions tensors of the type float 32
  # the shape indicates that the first three dimenisons, height, width sequence lenght, are variable
  # the last dimensions is fixed at 3, corresponding to the RGB color channels.
  
  The second part indicates that the labels yielded by the generator will be scalar values (shape = ()) of type int16
  """
  output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                      tf.TensorSpec(shape = (), dtype = tf.int16))

  # Converts the generator into a TensorFlow dataset
  train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 10, training=True),
                                            output_signature = output_signature)

  # Create the validation set
  val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], 10),
                                          output_signature = output_signature)

  AUTOTUNE = tf.data.AUTOTUNE # Instructs TensorFlow to dynamically adjust the number of elements to prefetch in the background
  # optimzing laoding performance at runtime based on the system's current conditions

  # Caches the dataset after loading from disk, subsequent epochs iwll load data much faster by avoiding disk reads
  # Shuffles the dataset with a buffer size of 1000, reduces overfitting by ensuring that batches are not correlated.
  # Prefetches dataset elements in the background while training, ensures data loading does not become a bottleneck.

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
  val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

  # Applies batch processing to both the training and validation datasets, batch size of 2
  # .batch method is part of the TensorFlow's tf.data.Dataset API, used to combine consecutive datasets into batches,
  # It transforms the dataset so that instead of yielding individual elements, it yeilds batches of elements.
  train_ds = train_ds.batch(2)
  val_ds = val_ds.batch(2)

  train_frames, train_labels = next(iter(train_ds))
  print(f'Shape of training set of frames: {train_frames.shape}')
  print(f'Shape of training labels: {train_labels.shape}')

  val_frames, val_labels = next(iter(val_ds))
  print(f'Shape of validation set of frames: {val_frames.shape}')
  print(f'Shape of validation labels: {val_labels.shape}')

  # Loads the EfficientNetB0 model without its top (classification) layer, to be used as a feature extrator for frame images
  net = tf.keras.applications.EfficientNetB0(include_top = False)
  # Freezes the layers, making them non-trainable. This is done to reuse the pre-trained weights without modifying them during training
  # Allows us to focus on the layers added to for the specific task only
  net.trainable = False

  model = tf.keras.Sequential([ # Defined as Sequentai, arranging the layers in a linear stack
      tf.keras.layers.Rescaling(scale=255), # Scales the input pixel values ot the range expected [0, 255]
      tf.keras.layers.TimeDistributed(net), # Applies the net model to each frame/timestep indepently to extract features
      tf.keras.layers.Dense(10), # A fully connected layer for 10 classes to learn classification
      tf.keras.layers.GlobalAveragePooling3D() # Pools the features extracted from teh video over the time dimension as well
    # as the spatial dimensions, reduicng the output to a single vector per video for classification.
  ])

  # Model is complied with the Adam optimizer and Sparse Categorical Crossentropy loss function
  # The from logist = true parameter indicates that the output of the model is not normalized (model will output logits)
  # Accuracy is used as metric for evaluation
  model.compile(optimizer = 'adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics=['accuracy'])
  # Model is trained using the train_ds, val_ds as validation, running for 10 epochs, but stopping early to prevent overfitting
  # The early stopping callback monitors val_loss, stopping training if it does not improve for 2 consecutive epochs.
  model.fit(train_ds,
            epochs = 10,
            validation_data = val_ds,
            callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'))
  main()