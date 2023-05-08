import tensorflow as tf
import tqdm
import os
import argparse
import nibabel as nib
import numpy as np
from samseg.VAE import VAE
from samseg.VAE_utils import *
from datetime import datetime as dt

#################################################################
# Train a Variational Autoencoder to learn segmentation shapes
#
# This script should be run after create_dataset.py
# 
#################################################################

# 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--out-dir', type=str, help="Directory where to store output results", required=True)
parser.add_argument('--data-dir', type=str, help="Directory where training and test datasets are stored.", required=True)
parser.add_argument('--save-probs', type=bool, default=True, help="Save individual sampled probabilities.")
parser.add_argument('--save-reconstruction', type=bool, default=True, help="Save reconstructed images.")
parser.add_argument('--augmentation', type=bool, default=True, help="Augment images by flipping axes and rotation.")
parser.add_argument('--max-angle', type=float, default=10.0, help="Max angle of rotation along each axis.")
parser.add_argument('--spatial-dir', help="Directory where training and test spatial prior data are stored.")
parser.add_argument('--batch-size', type=int, default=10, help="Batch size.")
parser.add_argument('--learning-rate', type=float, default=0.0001, help="Learning rate.")
parser.add_argument('--epochs', type=int, default=500, help="Number of epochs.")
parser.add_argument('--fine-tuning', action='store_true', default=False, help="Enable fine tuning of a model.")
parser.add_argument('--fine-tuning-model', type=str, help="Path to VAE model to fine tune.")
parser.add_argument('--name', type=str, default="", help="Name of the experiment.")
parser.add_argument('--clip', type=float, default=1.0, help="Value to clip gradients.")
parser.add_argument('--seed', type=int, default=12345, help="Seed number") 
args = parser.parse_args()

# Create random numpy object with given seed, and pass it around to other functions in other files
np_rng = np.random.default_rng(args.seed)

# Create experiment name
date_str = str(dt.now())[:-7].replace(":", "-").replace(" ", "-")
date_str += "-lr=%.5f"%args.learning_rate
if args.name != "":
    date_str += "-%s"%args.name

#
models_dir = os.path.join(args.out_dir, date_str, 'models')
samples_dir = os.path.join(args.out_dir, date_str, 'images')
log_dir = os.path.join(args.out_dir, date_str, 'logs')

os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

if args.save_reconstruction:
    os.makedirs(os.path.join(samples_dir, 'reconstructions'), exist_ok=True)
if args.save_probs:
    os.makedirs(os.path.join(samples_dir, 'probabilities'), exist_ok=True)

writer = tf.summary.create_file_writer(log_dir)

# Load data
train_filename = os.path.join(args.data_dir, 'train_images.npz')
test_filename = os.path.join(args.data_dir, 'test_images.npz')

print('Loading data... ')
x_train = np.load(train_filename)["data"]
x_test = np.load(test_filename)["data"]
batch_train = x_train.shape[0] // args.batch_size
batch_test = x_test.shape[0] // args.batch_size
if batch_test == 0:
    batch_test = 1
print("Data loaded")

# Loading spatial images
use_spatial_prior = args.spatial_dir is not None
if use_spatial_prior:
    print("Loading spatial data... ")
    spatial_data_train = np.load(os.path.join(args.spatial_dir, 'train_images_spatial.npz'))["data"]
    spatial_data_test = np.load(os.path.join(args.spatial_dir, 'train_images_spatial.npz'))["data"]
    print("Spatial data loaded")
else:
    spatial_data_train = None
    spatial_data_test = None


width, height, depth, num_classes = x_train.shape[1:]

print("Data shape")
print("Width: " + str(width) + " Height: " + str(height) + " Depth: " + str(depth) + " Num classes: " + str(num_classes))

# Define some variables
number_of_reconstructions = 1
number_of_samples = 1

# Save all VAE info in a file for future usage
np.savez(os.path.join(args.out_dir, date_str, 'VAE_info'), width=width, height=height, depth=depth,
         num_classes=num_classes)

# Define the model
model = VAE(width, height, depth, num_classes=num_classes, use_spatial_weights=use_spatial_prior)
model.global_step = 0

if args.fine_tuning:
    print("Loading fine tuned model")
    model.encode(x=np.zeros([1, width, height, depth, num_classes]))
    model.load_weights(os.path.join(args.fine_tuning_model))
    print("Model loaded")

model.printSummary()

optimizer = tf.keras.optimizers.Adam(args.learning_rate)

print('Start training...')
for epoch in range(model.global_step, model.global_step + args.epochs):

    # Train
    loss = 0
    rec_loss = 0
    kl_loss = 0

    for train_idx in tqdm.tqdm(range(int(batch_train))):
        feed = x_train[train_idx * args.batch_size:(train_idx + 1) * args.batch_size].astype(np.float32)
        if args.spatial_dir is not None:
            feed_spatial = spatial_data_train[train_idx * args.batch_size:(train_idx + 1) * args.batch_size].astype(np.float32)
        else:
            feed_spatial = None
        if args.augmentation:
            feed, feed_spatial = augment_batch(feed, args.max_angle, np_rng, feed_spatial)
        loss_it, rec_loss_it, kl_loss_it = compute_apply_gradients(model, feed, optimizer, clip_val=args.clip,
                                                                   spatial_weights=feed_spatial,
                                                                   num_classes=num_classes, seed=args.seed)
        loss += loss_it
        rec_loss += rec_loss_it
        kl_loss += kl_loss_it
    print('Epoch: {}, Train set loss: {}, rec: {}, KL: {}'.format(epoch,
                                                                  loss / batch_train,
                                                                  rec_loss / batch_train,
                                                                  kl_loss / batch_train))
    with writer.as_default():
        tf.summary.scalar('train losses/loss', loss / batch_train, step=model.global_step)
        tf.summary.scalar('train losses/rec_loss', rec_loss / batch_train, step=model.global_step)
        tf.summary.scalar('train losses/KL_loss',  kl_loss / batch_train, step=model.global_step)

    model.global_step += 1

    if epoch%50 == 0:
        model.save_weights(os.path.join(models_dir, 'model_epoch_%d.h5' % (epoch)))
        print("Model saved")
        print("Generate samples...")
        if args.save_reconstruction:
            # First from test data
            rand_idx = np_rng.integers(x_test.shape[0])
            feed = x_test[rand_idx:rand_idx + number_of_reconstructions].astype(np.float32)
            if args.spatial_dir is not None:
                feed_spatial = spatial_data_test[rand_idx:rand_idx + number_of_reconstructions]
            else:
                feed_spatial = None
            reconstructed = np.array(reconstruct(model, feed, spatial_weights=feed_spatial))
            if num_classes != 1:
                feed = np.argmax(feed, axis=-1).astype(float)
            for i in range(number_of_reconstructions):
                img = nib.Nifti1Image(reconstructed[i], np.eye(4))
                nib.save(img, os.path.join(samples_dir, 'reconstructions', 'ep_' + str(epoch) + 'rec_test_' + str(i) + '.nii.gz'))
                img = nib.Nifti1Image(feed[i], np.eye(4))
                nib.save(img, os.path.join(samples_dir, 'reconstructions', 'ep_' + str(epoch) + 'orig_test_' + str(i) + '.nii.gz'))
            # Then from training data
            rand_idx = np_rng.integers(x_train.shape[0])
            feed = x_train[rand_idx:rand_idx + number_of_reconstructions].astype(np.float32)
            if args.spatial_dir is not None:
                feed_spatial = spatial_data_test[rand_idx:rand_idx + number_of_reconstructions]
            else:
                feed_spatial = None
            reconstructed = np.array(reconstruct(model, feed, spatial_weights=feed_spatial, seed=args.seed))
            if num_classes != 1:
                feed = np.argmax(feed, axis=-1).astype(float)
            for i in range(number_of_reconstructions):
                img = nib.Nifti1Image(reconstructed[i], np.eye(4))
                nib.save(img, os.path.join(samples_dir, 'reconstructions', 'ep_' + str(epoch) + 'rec_train_' + str(i) + '.nii.gz'))
                img = nib.Nifti1Image(feed[i], np.eye(4))
                nib.save(img, os.path.join(samples_dir, 'reconstructions', 'ep_' + str(epoch) + 'orig_train_' + str(i) + '.nii.gz'))

        if args.spatial_dir is not None:
            # Here we are feeding the first training data samples for generation
            feed_spatial = spatial_data_train[0:number_of_samples, :, :, :]
        else:
            feed_spatial = None

        generated = model.sample(samples=number_of_samples, spatial_weights=feed_spatial, seed=args.seed)
        segmentation = np.argmax(generated, axis=-1).astype(float)
        for i in range(number_of_samples):
            img = nib.Nifti1Image(segmentation[i], np.eye(4))
            nib.save(img, os.path.join(samples_dir, 'ep_' + str(epoch) + 'sample_' + str(i) + '.nii.gz'))
            if args.save_probs:
                img = nib.Nifti1Image(np.array(generated[i]), np.eye(4))
                nib.save(img, os.path.join(samples_dir, 'probabilities', 'ep_' + str(epoch) + '_' + str(i) + '.nii.gz'))

    # Test
    loss = 0
    rec_loss = 0
    kl_loss = 0
    for test_idx in tqdm.tqdm(range(int(batch_test))):
        if batch_test != 1:
            feed = x_test[test_idx * args.batch_size:(test_idx + 1) * args.batch_size].astype(np.float32)
        else:
            feed = np.resize(x_test.astype(np.float32), [args.batch_size, x_test.shape[1], x_test.shape[2], x_test.shape[3], 1])
        if args.spatial_dir is not None:
            if batch_test != 1:
                feed_spatial = spatial_data_test[test_idx * args.batch_size:(test_idx + 1) * args.batch_size].astype(np.float32)
            else:
                feed_spatial = np.resize(spatial_data_test.astype(np.float32), [args.batch_size, feed_spatial.shape[1], feed_spatial.shape[2], feed_spatial.shape[3], 1])
        else:
            feed_spatial = None
        loss_it, rec_loss_it, kl_loss_it = compute_loss(model, feed, spatial_weights=feed_spatial,
                                                        num_classes=num_classes, seed=args.seed)
        loss += loss_it
        rec_loss += rec_loss_it
        kl_loss += kl_loss_it
    print('Epoch: {}, Test set loss: {}, rec: {}, KL: {}'.format(epoch,
                                                                 loss / batch_test,
                                                                 rec_loss / batch_test,
                                                                 kl_loss / batch_test))
        tf.summary.scalar('test losses/loss', loss / batch_test, step=model.global_step)
        tf.summary.scalar('test losses/rec_loss',  rec_loss / batch_test, step=model.global_step)
        tf.summary.scalar('test losses/KL_loss',  kl_loss / batch_test, step=model.global_step)

    # Shuffle the train dataset each epoch
    np_rng.shuffle(x_train)

if args.save_reconstruction:
    rand_idx = np_rng.integers(x_test.shape[0] - number_of_reconstructions)
    feed = x_test[rand_idx:rand_idx + number_of_reconstructions].astype(np.float32)
    if args.spatial_dir is not None:
        feed_spatial = spatial_data_test[rand_idx:rand_idx + number_of_reconstructions]
    else:
        feed_spatial = None
    reconstructed = np.argmax(reconstruct(model, feed, spatial_weights=feed_spatial, seed=args.seed), axis=-1).astype(float)
    if num_classes != 1:
        feed = np.argmax(feed, axis=-1).astype(float)
    for i in range(number_of_reconstructions):
        img = nib.Nifti1Image(reconstructed[i], np.eye(4))
        nib.save(img, os.path.join(samples_dir, 'reconstructions', 'end_rec_' + str(i) + '.nii.gz'))
        img = nib.Nifti1Image(feed[i], np.eye(4))
        nib.save(img, os.path.join(samples_dir, 'reconstructions' + 'end_orig_' + str(i) + '.nii.gz'))

print("Generate samples...")

if args.spatial_dir is not None:
    # Here we are feeding the first training data samples for generation
    feed_spatial = spatial_data_train[0:number_of_samples, :, :, :]
else:
    feed_spatial = None

generated = model.sample(samples=number_of_samples, spatial_weights=feed_spatial, seed=args.seed)
segmentation = np.argmax(generated, axis=-1).astype(float)
for i in range(number_of_samples):
    img = nib.Nifti1Image(segmentation[i], np.eye(4))
    nib.save(img, os.path.join(samples_dir, 'sample_end' + str(i) + '.nii.gz'))
    if args.save_probs:
        img = nib.Nifti1Image(np.array(generated[i]), np.eye(4))
        nib.save(img, os.path.join(samples_dir, 'probabilities', 'end_' + str(i) + '.nii.gz'))

# Save the model to disk.
print("Final model saved")
model.save_weights(os.path.join(models_dir, 'model.h5'))
print("Train finished")



