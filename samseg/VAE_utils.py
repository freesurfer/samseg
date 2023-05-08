import tensorflow as tf
import numpy as np
import scipy

@tf.function
def compute_loss(model, x, training=False, target=None, spatial_weights=None, num_classes=1, seed=12345):
    mean, logvar = model.encode(x, training=training)
    z = model.reparameterize(mean, logvar, seed=seed)
    x_logits = model.decode(z, training=training, spatial_weights=spatial_weights)
    if target is None:
        target = x

    if num_classes == 1:
        cross_ent = tf.keras.losses.binary_crossentropy(y_true=target, y_pred=x_logits)
    else:
        cross_ent = tf.keras.losses.categorical_crossentropy(y_true=target, y_pred=x_logits)

    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))

    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar)), axis=[1]))

    return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss



@tf.function
def compute_apply_gradients(model, x, optimizer, target=None, clip_val=1, spatial_weights=None, num_classes=1, seed=12345):
    with tf.GradientTape() as tape:
        loss, reconstruction_loss, kl_loss = compute_loss(model, x, training=True, target=target,
                                                          spatial_weights=spatial_weights, num_classes=num_classes, seed=seed)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_value(g, -clip_val, clip_val) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, reconstruction_loss, kl_loss


@tf.function
def reconstruct(model, x, spatial_weights=None, seed=12345):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar, seed=seed)
    x_logits = model.decode(z, spatial_weights=spatial_weights)
    return x_logits


# Augment training batch
def augment_batch(x, max_angle, np_rng, spatial_priors=None):
    # First we apply random rotation
    x, spatial_priors = random_rotation(x, max_angle, spatial_priors, np_rng)
    # Then we flip left and right hemisphere
    i = np_rng.integers(2)
    if i==0:
        return x, spatial_priors
    if i==1:
        if spatial_priors is not None:
            return x[:, ::-1, :, :, :], spatial_priors[:, ::-1, :, :, :]
        else:
            return x[:, ::-1, :, :, :], spatial_priors


# Randomly rotate an image by a random angle (-max_angle, max_angle)
def random_rotation(batch, max_angle, spatial_priors, np_rng):

    #
    batch_rot = np.zeros(batch.shape)
    if spatial_priors is not None:
        spatial_priors_rot = np.zeros(spatial_priors.shape)

    for i in range(batch.shape[0]):

        if np_rng.choice([0, 1]):

            j = np_rng.integers(3)
            if j == 0:
                # rotate along z-axis
                angle = np_rng.uniform(-max_angle, max_angle)
                batch_rot[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, mode='reflect', axes=(0, 1), reshape=False, order=0)
                if spatial_priors is not None:
                    spatial_priors_rot[i] = scipy.ndimage.interpolation.rotate(spatial_priors[i], angle, mode='reflect', axes=(0, 1), reshape=False, order=0)
            elif j == 1:
                # rotate along y-axis
                angle = np_rng.uniform(-max_angle, max_angle)
                batch_rot[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, mode='reflect', axes=(0, 2), reshape=False, order=0)
                if spatial_priors is not None:
                    spatial_priors_rot[i] = scipy.ndimage.interpolation.rotate(spatial_priors[i], angle, mode='reflect', axes=(0, 2), reshape=False, order=0)

            else:
                # rotate along x-axis
                angle = np_rng.uniform(-max_angle, max_angle)
                batch_rot[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, mode='reflect', axes=(1, 2), reshape=False, order=0)
                if spatial_priors is not None:
                    spatial_priors_rot[i] = scipy.ndimage.interpolation.rotate(spatial_priors[i], angle, mode='reflect', axes=(1, 2), reshape=False, order=0)

        else:
            batch_rot[i] = batch[i]
            if spatial_priors is not None:
                spatial_priors_rot[i] = spatial_priors[i]

    if spatial_priors is not None:
        return batch_rot, spatial_priors_rot
    else:
        return batch_rot, spatial_priors


