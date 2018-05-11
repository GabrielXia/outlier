import numpy as np
from tensorflow.python.platform import gfile
import gzip
import os
from PIL import Image

import outlier_data

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 np array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 np array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, num_classes=10):
  """Extract the labels into a 1D uint8 np array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 np array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def extract_if_outlier(f):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2053:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def get_outlier_with_ratio(ratio, root):
    ratio = int(ratio * 10) / 10.0
    dir = 'mnist_outlier_' + str(ratio)
    dir = os.path.join(root, dir)
    images_gz = dir + '/images-' + str(ratio) + '-ubyte.gz'
    labels_outlier = dir + '/labels-outlier-' + str(ratio) + '-ubyte.gz'
    labels_true = dir + '/labels-true-' + str(ratio) + '-ubyte.gz'
    with gfile.Open(images_gz, 'rb') as f:
      train_images = extract_images(f)
    with gfile.Open(labels_outlier, 'rb') as f:
      train_labels = extract_labels(f)
    with gfile.Open(labels_true, 'rb') as f:
      train_if_outlier = extract_if_outlier(f)
    return int_to_float(train_images), train_labels, train_if_outlier

def get_test(root):
    images_gz = 'test-images-ubyte.gz'
    labels_gz = 'test-labels-ubyte.gz'
    images_gz = os.path.join(root, images_gz)
    labels_gz = os.path.join(root, labels_gz)
    with gfile.Open(images_gz, 'rb') as f:
      test_images = extract_images(f)
    with gfile.Open(labels_gz, 'rb') as f:
      test_labels = extract_labels(f)
    return int_to_float(test_images), test_labels


def get_validation(root):
    images_gz = 'validation-images-ubyte.gz'
    labels_gz = 'validation-labels-ubyte.gz'
    images_gz = os.path.join(root, images_gz)
    labels_gz = os.path.join(root, labels_gz)
    with gfile.Open(images_gz, 'rb') as f:
      validation_images = extract_images(f)
    with gfile.Open(labels_gz, 'rb') as f:
      validation_labels = extract_labels(f)
    return int_to_float(validation_images), validation_labels


def int_to_float(images):
    #imagedata = np.zeros_like(images, dtype=np.uint8)
    images = images.astype(np.float32)
    return np.multiply(images, 1.0 / 255.0)


class MnistOutlier(outlier_data.Mislabeled):
    def __init__(self, outlier_ratio, dir='../data/mnist/',
                 transform = None, target_transform = None):
        self.root = os.path.expanduser(dir)
        self.outlier_ratio = outlier_ratio
        self.train_images, self.train_labels, self.true_label = get_outlier_with_ratio(outlier_ratio, self.root)
        #self.test_images, self.test_labels = get_test(self.root)
        #self.validation_images, self.validation_labels = get_validation(self.root)
        self.if_outlier = (self.train_labels != self.true_label).astype(int)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target, if_outlier, true_label = self.train_images[index], self.train_labels[index], \
                                              self.if_outlier[index], self.true_label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, if_outlier, true_label

    def __len__(self):
        return len(self.train_labels)

    def __getratio__(self):
        return self.outlier_ratio

    def __get_if_outlier__(self, index):
        return self.if_outlier[index]

    def __get_true_label__(self, index):
        return self.true_label[index]

if __name__ == '__main__':
    a = MnistOutlier(0.1)