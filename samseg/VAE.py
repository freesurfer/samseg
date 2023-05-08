import tensorflow as tf
import tensorflow_probability as tfp

#
# VAE class:
# 2 convolutional layers for encoder and 2 convolutional layers for decoder
# 
# If spatial weights are provided, a voxel-wise bias is added to the last layer of the decoder, before the sigmoid/softmax activation function 
# 
class VAE(tf.keras.Model):
    def __init__(self, width, height, depth, num_classes=1, alpha=1.0, use_spatial_weights=False):
        super(VAE, self).__init__()

        self.width = width
        self.height = height
        self.depth = depth
        self.num_classes = num_classes
        self.alpha = alpha
        self.use_spatial_weights = use_spatial_weights

        # Inference/Encoder net
        self.inference_net = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.width, self.height, self.depth, self.num_classes)),

                tf.keras.layers.Conv3D(filters=32, kernel_size=[3, 3, 3], strides=[2, 2, 2],
                                       padding='SAME', activation=tf.nn.leaky_relu),

                tf.keras.layers.Conv3D(filters=32, kernel_size=[3, 3, 3], strides=[2, 2, 2],
                                       padding='SAME')
            ]
        )

        # Compute latent dim shape from input shape
        self.latent_width = tf.cast(tf.math.ceil(tf.math.ceil(self.width / 2) / 2), dtype=tf.int32)
        self.latent_height = tf.cast(tf.math.ceil(tf.math.ceil(self.height / 2) / 2), dtype=tf.int32)
        self.latent_depth = tf.cast(tf.math.ceil(tf.math.ceil(self.depth / 2) / 2), dtype=tf.int32)
        self.latent_filters = tf.cast(16, dtype=tf.int32)
        self.latent_dim_shape = tf.shape(tf.ones((self.latent_width, self.latent_height, self.latent_depth, self.latent_filters)))

        # Generative/Decoder net
        self.generative_net = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.latent_dim_shape),

                tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=[3, 3, 3], strides=[2, 2, 2],
                                       padding='SAME', activation=tf.nn.leaky_relu),

                tf.keras.layers.Conv3DTranspose(filters=self.num_classes, kernel_size=[3, 3, 3], strides=[2, 2, 2],
                                       padding='SAME', use_bias=not self.use_spatial_weights),
            ]

        )

    @tf.function
    def sample(self, eps=None, samples=1, sample_decode=False, spatial_weights=None, seed=12345):
        if eps is None:
            eps = tf.random.normal(shape=(samples, self.latent_width, self.latent_height, self.latent_depth, self.latent_filters), seed=seed)
        return self.decode(eps, sample_decode=sample_decode, spatial_weights=spatial_weights)


    def encode(self, x, training=False):
        mean, logvar = tf.split(self.inference_net(x, training=training), num_or_size_splits=2, axis=-1)
        return mean, logvar


    def reparameterize(self, mean, logvar, seed=12345):
        eps = tf.random.normal(shape=mean.shape, seed=seed)
        return eps * tf.exp(logvar * .5) + mean


    def decode(self, z, training=False, sample_decode=False, spatial_weights=None):
        logits = self.generative_net(z, training=training)

        # scale logits to desired shape
        logits = self.pad_to_proper_size(logits)

        # Add spatial weights (in log domain) to the logits
        # Alpha is a factor deciding how much these weights are driving the image generation
        if self.use_spatial_weights and spatial_weights is not None:
            logs = tf.cast(tf.math.log(spatial_weights + 1e-20), tf.float32)
            logits = logits + self.alpha * logs

        if sample_decode:
            if self.num_classes == 1:
                return tfp.distributions.Bernoulli(logits=logits).sample()
            else:
                return tfp.distributions.Categorical(logits=logits).sample()

        if self.num_classes == 1:
            logits = tf.nn.sigmoid(logits)
        else:
            logits = tf.nn.softmax(logits)

        return logits


    def printSummary(self):
        self.inference_net.summary()
        self.generative_net.summary()


    # Automatically pad logits to input size
    def pad_to_proper_size(self, logits):
        shape = logits.shape
        quot1, rem1 = divmod(shape[1] - self.width, 2)
        quot2, rem2 = divmod(shape[2] - self.height, 2)
        quot3, rem3 = divmod(shape[3] - self.depth, 2)
        return logits[:, quot1:shape[1] - quot1 - rem1, quot2:shape[2] - quot2 - rem2, quot3:shape[3] - quot3 - rem3, :]

