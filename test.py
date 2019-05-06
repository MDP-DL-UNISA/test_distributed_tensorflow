from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os
import math
import numpy as np

width = 224
height = 224
channels = 3
n_inputs = width * height * channels
batch_size = 1
num_epoch = 2
FLAGS = None
samples_path = "/home/simoneneurone/Documents/images/train/"

def preprocess_image(img):
    """ It reads an image, it resize it to have the lowest dimesnion of 256px,
		it randomly choose a 224x224 crop inside the resized image and normilize the numpy
		array subtracting the ImageNet training set mean
		Args:
			images_path: path of the image
		Returns:
			cropped_im_array: the numpy array of the image normalized [width, height, channels]
	"""
    IMAGENET_MEAN = [123.68, 116.779, 103.939]  # rgb format

    img = tf.image.resize_images(img, (width, height))
    img = tf.random_crop(img, [width, height, channels])
    img = tf.subtract(img, IMAGENET_MEAN)
    img = tf.reshape(img, [-1])
    return img


def load_dataset_filenames(base_path, n_workers):
    labels = []
    filenames = []
    labels_name = []
    dirs = os.listdir(base_path)
    n_labels = len(dirs)
    lab = 0
    for name in dirs:
        current_path = base_path + name
        files = os.listdir(current_path)
        tot_images_in_dir = len(files)
        if FLAGS.task_index == 0:
            files = files[0:int(math.floor(tot_images_in_dir / n_workers))]
        elif FLAGS.task_index == n_workers - 1:
            files = files[int(FLAGS.task_index * math.floor(tot_images_in_dir / n_workers)):tot_images_in_dir]
        else:
            files = files[int(FLAGS.task_index * math.floor(tot_images_in_dir / n_workers)):int(
                (FLAGS.task_index + 1) * math.floor(tot_images_in_dir / n_workers))]
        for file in files:
            labels_name.append(name)
            labels.append(lab)
            filenames.append(current_path + "/" + file)
        lab += 1
    one_hot = []
    for i in range(len(labels)):
        ll = np.zeros(n_labels, dtype=int)
        ll[labels[i]] = 1
        one_hot.append(ll.tolist())
    return filenames, one_hot, labels_name, n_labels


def load_dataset(n_workers):
    print("Loading filenames...")
    filenames, labels, labels_name, n_labels = load_dataset_filenames(samples_path, n_workers)
    num_samples = len(labels)
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    print("Build dataset ...")
    # step 2: create a dataset returning slices of `filenames`
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # step 3: parse every image in the dataset using `map`
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        imag = preprocess_image(image)
        return imag, label

    dataset = dataset.map(_parse_function).repeat(num_epoch)
    dataset = dataset.batch(batch_size)
    # step 4: create iterator and final input tensor
    iterator = dataset.make_one_shot_iterator()

    images, labels = iterator.get_next()
    return images, labels, num_samples, n_labels


def main():
    # Configure
    config = tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.allow_growth = True
    # Server Setup
    cluster_spec = {
        'ps': ['172.16.69.190:2222'],
        'worker': ['172.16.69.185:2223']
    }  # allows this node know about all other nodes
    n_pss = len(cluster_spec['ps'])  # the number of parameter servers
    n_workers = len(cluster_spec['worker'])  # the number of worker nodes
    cluster = tf.train.ClusterSpec(cluster_spec)

    if FLAGS.job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=FLAGS.task_index,
                                 config=config)
        server.join()
    else:  # it must be a worker server
        images, labels, num_samples, n_classes = load_dataset(n_workers)
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=FLAGS.task_index)
        # Monitored Training Session
        sess = tf.train.MonitoredTrainingSession(master=server.target, config=config)

        print("Coming......")
        try:
            while True:
                im, ll = sess.run([images, labels])
                print(im.shape)
                print(ll.shape)

        except tf.errors.OutOfRangeError:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS.task_index)
    main()
