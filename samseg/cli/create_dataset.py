import os
import numpy as np
import nibabel as nib
import argparse

#####################################
#
# Create dataset folder with one hot representation of input data
#
# Dataset is split into training and test set according to --test-fraction
#
# Strong assumption, but needed: all the segmentations are in the same image space and have the same image size
#
#####################################

# Given a segmentation "x" and a list of segmentation labels "seg_labels", return a one hot representation of "x". 
# Note that the order of the labels is given by the order of the list "seg_labels"
def voxel_one_hot(x, seg_labels):
    number_of_classes = len(seg_labels)
    x_new = np.zeros(list(x.shape) + [number_of_classes], dtype=bool)
    for c in range(number_of_classes):
        x_new[:, :, :, c] = x == seg_labels[c]
    return x_new


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, help="Path to training data.", required=True)
parser.add_argument('--out-dir', type=str, help="Output directory.", required=True)
parser.add_argument('--filename', help="File name (or common substring) of the training segmentations.")
parser.add_argument('--all-in-one-dir', action='store_true', default=False, help="All training data is store in one directory (no sub-directories). Default is one sub-directory for each subject.")
parser.add_argument('--test-fraction', type=float, default=0.2, help="Fraction of training samples allocated for testing.")
parser.add_argument('--save-average', action='store_true', default=False, help="Save average of training data. To be used for template registration")
parser.add_argument('--path-spatial-priors', type=str, help="Path to deformed spatial priors.")
parser.add_argument('--random-seed', type=int, default=12345, help='Random seed.')
args = parser.parse_args()

# Create output directory, if it doesn't exist yet
os.makedirs(args.out_dir, exist_ok=True)

# Load subjects
subjects = os.listdir(args.path)
subjects.sort()
print("Found " + str(len(subjects)) + " subjects")
print("Loading data...")

if not args.all_in_one_dir:

    if args.filename is not None:
        dir_list = os.listdir(os.path.join(args.path, subjects[0]))
        filename = next((s for s in dir_list if args.filename in s), None)
        tmp = nib.load(os.path.join(args.path, subjects[0], filename))
    else:
        # Get file name in subject dir
        filename = os.listdir(os.path.join(args.path, subjects[0]))
        tmp = nib.load(filename)
else:
    tmp = nib.load(os.path.join(args.path, subjects[0]))

width, height, depth = tmp.shape
affine = tmp.affine

data = np.zeros([len(subjects), width, height, depth], dtype=np.int8)
for i, subject in enumerate(subjects):
    print("Loading: " + str(subject))
    if not args.all_in_one_dir:
        if args.filename is not None:
            dir_list = os.listdir(os.path.join(args.path, subject))
            filename = next((s for s in dir_list if args.filename in s), None)
            subject_file_name = os.path.join(args.path, subject, filename)
        else:
            # Get file name in subject dir
            subject_file_name = os.listdir(os.path.join(args.path, subject))
    else:
        subject_file_name = os.path.join(args.path, subject)
    data[i] = nib.load(subject_file_name).get_fdata()

#
print("Computing unique values in the data")
unique_values = np.unique(data)
if len(unique_values) == 2: # We have a binary classification
    seg = np.array(data, dtype=bool)
    # Add extra dimension on the last axis, defining that we have only 1 class (either 0 or 1)
    seg = np.expand_dims(seg, axis=-1)
else:
    print("Transforming data to one-hot representation")
    seg = np.zeros([len(subjects), width, height, depth, len(unique_values)], dtype=bool)
    for i in range(len(subjects)):
        seg[i] = voxel_one_hot(data[i], unique_values)

rng_numpy = np.random.default_rng(args.random_seed)
randomIdxs = rng_numpy.permutation(len(seg))
trainIdxs = randomIdxs[int(args.test_fraction * len(seg)):]
testIdxs = randomIdxs[:int(args.test_fraction * len(seg))]

# Save training and test data
np.savez_compressed(os.path.join(args.out_dir, 'train_images.npz'), data=seg[trainIdxs])
np.savez_compressed(os.path.join(args.out_dir, 'test_images.npz'), data=seg[testIdxs])

if args.save_average:
    print("Saving average")
    # Save average, if requested. We might use it for learning the transformation from training data to template data
    # Exclude background, which we assume has index 0
    if len(unique_values) == 2:
        avg = np.mean(seg, axis=0)
    else:
        avg = np.mean(np.sum(seg[:, :, :, :, 1:], axis=4), axis=0)
    img = nib.Nifti1Image(avg, affine)
    nib.save(img, os.path.join(args.out_dir, 'average.nii.gz'))

if args.path_spatial_priors is not None:
    print("Loading spatial priors")

    subjects = os.listdir(args.path)
    subjects.sort()

    # This can be a huge array, better use float32, especially since Tensorflow is using 32bits anyway
    spatial_priors = np.zeros_like(seg, dtype=np.float32)

    for i, subject in enumerate(subjects):
        print("Loading: " + str(subject))
        if not args.all_in_one_dir:
            if args.filename is not None:
                dir_list = os.listdir(os.path.join(args.path_spatial_priors, subject))
                filename = next((s for s in dir_list if args.filename in s), None)
                subject_file_name = os.path.join(args.path_spatial_priors, subject, filename)
            else:
                # Get file name in subject dir
                subject_file_name = os.listdir(os.path.join(args.path_spatial_priors, subject))
        else:
            subject_file_name = os.path.join(args.path_spatial_priors, subject)

        spatial_prior = nib.load(subject_file_name).get_fdata()
        if len(unique_values) == 2:
            spatial_priors[i, :, :, :, 0] = spatial_prior
        else:
            for class_number in range(len(unique_values) - 1):
                spatial_priors[i, :, :, :, class_number + 1] = spatial_prior[:, :, :, class_number]
            spatial_priors[i, :, :, :, 0] = 1 - np.sum(spatial_prior, axis=-1)

    # Save training and test data
    np.savez_compressed(os.path.join(args.out_dir, 'train_images_spatial.npz'), data=spatial_priors[trainIdxs])
    np.savez_compressed(os.path.join(args.out_dir, 'test_images_spatial.npz'), data=spatial_priors[testIdxs])

    if args.save_average:
        print("Saving average")
        # Save average
        # Exclude background, which we assume has index 0
        if len(unique_values) == 2:
            avg = np.mean(spatial_priors, axis=0)
        else:
            avg = np.mean(np.sum(spatial_priors[:, :, :, :, 1:], axis=4), axis=0)

        img = nib.Nifti1Image(avg, affine)
        nib.save(img, os.path.join(args.out_dir, 'average_spatial.nii.gz'))

print("Done")

