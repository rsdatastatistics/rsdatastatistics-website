+++
authors = ["Rauzan Sumara"]
title = "Implementation of GAN and cGAN Models"
slug = "implementation-of-gan-and-cgan"
date = "2025-12-22"
tags = [
    "GAN",
    "cGAN",
]
categories = [
    "Deep Learning",
]
image = "https://raw.githubusercontent.com/rauzansumara/gan-and-cgan/master/images/gan1.png"
+++


Here we are going to talk into the detail of what Generative Adversarial Network (GAN) and Conditional Generative Adversarial Network (cGAN) are. I will also explain and give an implementation both of them seperately. Code also can be obtained from my [**GitHub**](https://github.com/rauzansumara/gan-and-cgan/tree/main). This model is one of the most interesting ideas in computer science today. Two models are trained simultaneously by an adversarial process. A generator **("the artist")** learns to create images that look real, while a discriminator **("the art critic")** learns to tell real images apart from fakes. Thanks so much to [*Rowel Atienza*](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0), [*Jason Brownlee*](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/) and [*www.tensorflow.org*](https://www.tensorflow.org/tutorials/generative/dcgan) that inspired me from their articles. 

## Generative Adversarial Network (GAN)

Generative Adversarial Networks (GAN) is one of the most promising recent developments in Deep Learning. GAN, introduced by [Ian Goodfellow and friends](https://arxiv.org/abs/1406.2661) in 2014, attacks the problem of unsupervised learning by training two deep networks, called **Generator** and **Discriminator**, that compete and cooperate with each other. In the course of training, both networks eventually learn how to perform their tasks.

Figure 1 : A diagram of a generator and discriminator

![A diagram of a generator and discriminator](https://raw.githubusercontent.com/rauzansumara/gan-and-cgan/master/images/gan1.png "A diagram of a generator and discriminator") 
*Source: https://www.tensorflow.org/*

While the idea of GAN is simple in theory, it is very difficult to build a model that works. In GAN, there are two deep networks coupled together making back propagation of gradients twice as challenging. Deep Convolutional GAN (DCGAN) is one of the models that demonstrated how to build a practical GAN that is able to learn by itself how to synthesize new images. 

In this article, we discuss how a working GAN can be built using Keras on Tensorflow 2.x backend. We will train a DCGAN to learn how to write handwritten digits on the MNIST dataset.

### Setup library


```python
import os
import PIL
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, Sequential, optimizers, losses
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import LeakyReLU, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from IPython import display
tf.__version__
```




    '2.4.0'



### Load the dataset

We will use the MNIST dataset to train the **generator** and the **discriminator**. The generator will generate handwritten digits resembling the MNIST data.


```python
(train_images, train_labels), (_, _) = datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

### The Discriminator model

A discriminator that tells how real image is, is basically a deep Convolutional Neural Network (CNN) as shown in Figure 2. The discriminator model takes as input one 28×28 grayscale image and outputs a binary prediction as to whether the image is real (class=1) or fake (class=0). It is implemented as a modest convolutional neural network using best practices for GAN design such as using the LeakyReLU activation function, using a 2×2 stride to downsample. The activation function used in each CNN layer is a leaky ReLU. A dropout between 0.3 and 0.3 between layers prevent over fitting and memorization.

Figure 2 : The Discriminator

![The discriminator](https://raw.githubusercontent.com/rauzansumara/gan-and-cgan/master/images/Discriminator.png "The discriminator") 

*Source:* [*Rowel Atienza*](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)





```python
def make_discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model
```

### The Generator model

The generator synthesizes fake images. In Figure 3, the fake image is generated from a 100-dimensional noise (uniform distribution between -1.0 to 1.0) using the inverse of convolution, called transposed convolution. Instead of fractionally-strided convolution as suggested in DCGAN, upsampling between the first three layers is used since it synthesizes more realistic handwriting images. In between layers, batch normalization stabilizes learning. The activation function after each layer is a LeakyReLU. The output of the tanh at the last layer produces the fake image. 

Figure 3 : The Generator

![The Generator](https://raw.githubusercontent.com/rauzansumara/gan-and-cgan/master/images/Generator.png "The Generator") 

*Source:* [*Rowel Atienza*](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)



```python
def make_generator_model():
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
```

### Define the loss and optimizers

Define loss functions and optimizers for both models.


```python
# Define the loss and optimizers
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)
```

### Training 

So far, there are no models yet. It is time to build the models for training. We need two models: 1) Discriminator Model  and 2) Generator-Discriminator model. The generator-discriminator stacked together as shown in Figure 4. The Generator part is trying to fool the Discriminator and learning from its feedback at the same time. The training parameters are the same as in the Discriminator model except for a reduced learning rate and corresponding weight decay.

Training is the hardest part. We determine first if Discriminator model is correct by training it alone with real and fake images. Afterwards, the Discriminator and Generator-Discriminator model are trained one after the other.

Figure 4 : The Training Process

![The Training Process](https://raw.githubusercontent.com/rauzansumara/gan-and-cgan/master/images/gan2.png "The Training Process")

*Source:* [*Rowel Atienza*](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)



```python
# Define the training loop
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# Define generator and discriminator
generator = make_generator_model()
discriminator = make_discriminator_model()

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Training steps
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

# Generate and save images
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
```

Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about 3 minutes / epoch with the default settings on Colab.


```python
# This notebook also demonstrates how to save and restore models, 
# which can be helpful in case a long running training task is interrupted.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

train(train_dataset, EPOCHS) # train process

# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS) # display created images
```


    
![svg](/static/post/Impelentation_of_GAN_and_cGAN/output_17_0.svg)
    





    
![png](/static/post/Impelentation_of_GAN_and_cGAN/output_17_1.png)
    




```python
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

## Conditional Generative Adversarial Network (cGAN)

Generative Adversarial Networks, or GANs, are an architecture for training generative models, such as deep convolutional neural networks for generating images. Although GAN models are capable of generating new random plausible examples for a given dataset, there is no way to control the types of images that are generated other than trying to figure out the complex relationship between the latent space input to the generator and the generated images.

The conditional generative adversarial network, or cGAN for short, is a type of GAN that involves the conditional generation of images by a generator model. Image generation can be conditional on a class label, if available, allowing the targeted generated of images of a given type.

For example, the MNIST handwritten digit dataset has class labels of the corresponding integers, the CIFAR-10 small object photograph dataset has class labels for the corresponding objects in the photographs, and the Fashion-MNIST clothing dataset has class labels for the corresponding items of clothing.

There are two motivations for making use of the class label information in a GAN model.

 * Improve the GAN.
 * Targeted Image Generation.

Additional information that is correlated with the input images, such as class labels, can be used to improve the GAN. This improvement may come in the form of more stable training, faster training, and/or generated images that have better quality. Class labels can also be used for the deliberate or targeted generation of images of a given type.

Alternately, a GAN can be trained in such a way that both the generator and the discriminator models are conditioned on the class label. This means that when the trained generator model is used as a standalone model to generate images in the domain, images of a given type, or class label, can be generated.

The cGAN was first described by Mehdi Mirza and Simon Osindero in their 2014 paper titled [*“Conditional Generative Adversarial Nets.”*](https://arxiv.org/abs/1411.1784) In the paper, the authors motivate the approach based on the desire to direct the image generation process of the generator model.

Figure 5 : The cGAN

![The cGAN](https://raw.githubusercontent.com/rauzansumara/gan-and-cgan/master/images/cgan.png "The cGAN")

*Source:* [*Jason Brownlee*](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/)

And now we will also discuss how cGAN can be built using Keras on Tensorflow 2.x backend. We will train a the model to learn how to write handwritten digits on the MNIST dataset as well.



### Setup library


```python
# Import library
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from matplotlib import pyplot
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras import datasets, Sequential, optimizers, losses
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.layers import LeakyReLU, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model


```

### Load the dataset


```python
# example of loading the MNIST dataset
(trainX, trainy), (testX, testy) = datasets.mnist.load_data()

# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
```

    Train (60000, 28, 28) (60000,)
    Test (10000, 28, 28) (10000,)
    

### Define the discriminator model

The discriminator model takes as input one 28×28 grayscale image and outputs a binary prediction as to whether the image is real (class=1) or fake (class=0). It is implemented as a modest convolutional neural network using best practices for GAN design such as using the LeakyReLU activation function, using a 2×2 stride to downsample, and the adam version of stochastic gradient descent with a learning rate of 0.0001.

The *define_discriminator()* function below implements this, defining and compiling the discriminator model and returning it. The input shape of the image is parameterized as a default function argument in case you want to re-use the function for your own image data later.


```python
# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
    model = Sequential()

    # downsample
    model.add(Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # downsample
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # classifier
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    opt = optimizers.Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
```

### Define the generator model

The generator model takes as input a point in the latent space and outputs a single 28×28 grayscale image. This is achieved by using a fully connected layer to interpret the point in the latent space and provide sufficient activations that can be reshaped into many copies (in this case 128) of a low-resolution version of the output image (e.g. 7×7). This is then upsampled twice, doubling the size and quadrupling the area of the activations each time using transpose convolutional layers. The model uses best practices such as the LeakyReLU activation, a kernel size that is a factor of the stride size, and a hyperbolic tangent (tanh) activation function in the output layer.

The define_generator() function below defines the generator model, but intentionally does not compile it as it is not trained directly, then returns the model. The size of the latent space is parameterized as a function argument.


```python
# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()

    # foundation for 7x7 image
    n_nodes = 256 * 7 * 7
    model.add(Dense(n_nodes, use_bias=False, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))

    # upsample to 14x14
    model.add(Conv2DTranspose(64, (5,5), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # upsample to 28x28
    model.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # generate
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model
```

### The cGAN model

Next, a cGAN model can be defined that combines both the generator model and the discriminator model into one larger model. This larger model will be used to train the model weights in the generator, using the output and error calculated by the discriminator model. The discriminator model is trained separately, and as such, the model weights are marked as not trainable in this larger GAN model to ensure that only the weights of the generator model are updated. This change to the trainability of the discriminator weights only has an effect when training the combined GAN model, not when training the discriminator standalone.

This larger cGAN model takes as input a point in the latent space, uses the generator model to generate an image which is fed as input to the discriminator model, then is output or classified as real or fake.

The *define_gan()* function below implements this, taking the already-defined generator and discriminator models as input.


```python
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False

    # connect them
    model = Sequential()

    # add generator
    model.add(generator)

    # add the discriminator
    model.add(discriminator)
    
    # compile model
    opt = optimizers.Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
```

### Load and generate function

Now that we have defined the cGAN model, we need to train it. But, before we can train the model, we require input data.

* The first step is to load and prepare the Fashion MNIST dataset. We only require the images in the training dataset. The images are black and white, therefore we must add an additional channel dimension to transform them to be three dimensional, as expected by the convolutional layers of our models. Finally, the pixel values must be scaled to the range [-1,1] to match the output of the generator model. *The load_real_samples()* function below implements this, returning the loaded and scaled MNIST training dataset ready for modeling.

* We will require one batch (or a half) batch of real images from the dataset each update to the cGAN model. A simple way to achieve this is to select a random sample of images from the dataset each time. *The generate_real_samples()* function below implements this, taking the prepared dataset as an argument, selecting and returning a random sample of MNIST images and their corresponding class label for the discriminator, specifically class=1, indicating that they are real images.

* Next, we need inputs for the generator model. These are random points from the latent space, specifically Gaussian distributed random variables. *The generate_latent_points()* function implements this, taking the size of the latent space as an argument and the number of points required and returning them as a batch of input samples for the generator model.

* Next, we need to use the points in the latent space as input to the generator in order to generate new images. *The generate_fake_samples()* function below implements this, taking the generator model and size of the latent space as arguments, then generating points in the latent space and using them as input to the generator model. The function returns the generated images and their corresponding class label for the discriminator model, specifically class=0 to indicate they are fake or generated.


```python
# load real samples
def load_real_samples():
	# load dataset
	(trainX, _), (_, _) = datasets.mnist.load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n_samples, 1))
	return X, y
```

### Training model

We are now ready to fit the cGAN models.

The model is fit for 100 training epochs, which is arbitrary, as the model begins generating plausible handwriten digits after perhaps 20 epochs. A batch size of 128 samples is used, and each training epoch involves 60,000/128, or about 468 batches of real and fake samples and updates to the model.

First, the discriminator model is updated for a half batch of real samples, then a half batch of fake samples, together forming one batch of weight updates. The generator is then updated via the composite gan model. Importantly, the class label is set to 1 or real for the fake samples. This has the effect of updating the generator toward getting better at generating real samples on the next batch.

The *train()* function below implements this, taking the defined models, dataset, and size of the latent dimension as arguments and parameterizing the number of epochs and batch size with default arguments. The generator model is saved at the end of training.


```python
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# save the generator model
	g_model.save('generator.h5')
 
# size of the latent space
latent_dim = 16
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)

```

    
    >99, 458/468, d1=0.655, d2=0.662 g=0.804
    >99, 459/468, d1=0.639, d2=0.653 g=0.779
    >99, 460/468, d1=0.674, d2=0.618 g=0.819
    >99, 461/468, d1=0.634, d2=0.682 g=0.814
    >99, 462/468, d1=0.673, d2=0.676 g=0.788
    >99, 463/468, d1=0.644, d2=0.644 g=0.833
    >99, 464/468, d1=0.681, d2=0.655 g=0.823
    >99, 465/468, d1=0.660, d2=0.652 g=0.786
    >99, 466/468, d1=0.648, d2=0.606 g=0.781
    >99, 467/468, d1=0.645, d2=0.663 g=0.830
    >99, 468/468, d1=0.658, d2=0.626 g=0.784
    >100, 1/468, d1=0.691, d2=0.649 g=0.809
    >100, 2/468, d1=0.633, d2=0.653 g=0.810
    >100, 3/468, d1=0.693, d2=0.630 g=0.794
    >100, 4/468, d1=0.656, d2=0.686 g=0.799
    >100, 5/468, d1=0.696, d2=0.631 g=0.792
    >100, 6/468, d1=0.642, d2=0.691 g=0.757
    >100, 7/468, d1=0.677, d2=0.696 g=0.806
    >100, 8/468, d1=0.650, d2=0.627 g=0.811
    >100, 9/468, d1=0.673, d2=0.657 g=0.800
    >100, 10/468, d1=0.709, d2=0.618 g=0.779
    >100, 11/468, d1=0.645, d2=0.658 g=0.836
    >100, 12/468, d1=0.660, d2=0.645 g=0.837
    >100, 13/468, d1=0.689, d2=0.600 g=0.781
    >100, 14/468, d1=0.658, d2=0.662 g=0.818
    >100, 15/468, d1=0.657, d2=0.660 g=0.835
    >100, 16/468, d1=0.685, d2=0.674 g=0.793
    >100, 17/468, d1=0.724, d2=0.617 g=0.810
    >100, 18/468, d1=0.643, d2=0.668 g=0.751
    >100, 19/468, d1=0.674, d2=0.658 g=0.789
    >100, 20/468, d1=0.681, d2=0.690 g=0.820
    >100, 21/468, d1=0.710, d2=0.719 g=0.802
    >100, 22/468, d1=0.717, d2=0.698 g=0.756
    >100, 23/468, d1=0.748, d2=0.716 g=0.794
    >100, 24/468, d1=0.721, d2=0.667 g=0.780
    >100, 25/468, d1=0.636, d2=0.646 g=0.736
    >100, 26/468, d1=0.676, d2=0.729 g=0.782
    >100, 27/468, d1=0.634, d2=0.684 g=0.750
    >100, 28/468, d1=0.629, d2=0.684 g=0.770
    >100, 29/468, d1=0.685, d2=0.707 g=0.759
    >100, 30/468, d1=0.686, d2=0.677 g=0.761
    >100, 31/468, d1=0.690, d2=0.676 g=0.768
    >100, 32/468, d1=0.674, d2=0.672 g=0.743
    >100, 33/468, d1=0.619, d2=0.679 g=0.761
    >100, 34/468, d1=0.653, d2=0.659 g=0.766
    >100, 35/468, d1=0.660, d2=0.703 g=0.741
    >100, 36/468, d1=0.677, d2=0.678 g=0.768
    >100, 37/468, d1=0.629, d2=0.668 g=0.728
    >100, 38/468, d1=0.689, d2=0.705 g=0.727
    >100, 39/468, d1=0.674, d2=0.715 g=0.732
    >100, 40/468, d1=0.652, d2=0.713 g=0.710
    >100, 41/468, d1=0.636, d2=0.700 g=0.737
    >100, 42/468, d1=0.703, d2=0.725 g=0.714
    >100, 43/468, d1=0.670, d2=0.707 g=0.746
    >100, 44/468, d1=0.667, d2=0.709 g=0.733
    >100, 45/468, d1=0.685, d2=0.689 g=0.724
    >100, 46/468, d1=0.656, d2=0.695 g=0.754
    >100, 47/468, d1=0.675, d2=0.712 g=0.708
    >100, 48/468, d1=0.611, d2=0.682 g=0.732
    >100, 49/468, d1=0.684, d2=0.679 g=0.769
    >100, 50/468, d1=0.644, d2=0.731 g=0.767
    >100, 51/468, d1=0.676, d2=0.634 g=0.761
    >100, 52/468, d1=0.650, d2=0.639 g=0.767
    >100, 53/468, d1=0.648, d2=0.646 g=0.765
    >100, 54/468, d1=0.706, d2=0.691 g=0.785
    >100, 55/468, d1=0.712, d2=0.681 g=0.744
    >100, 56/468, d1=0.712, d2=0.673 g=0.761
    >100, 57/468, d1=0.690, d2=0.691 g=0.720
    >100, 58/468, d1=0.688, d2=0.736 g=0.786
    >100, 59/468, d1=0.616, d2=0.673 g=0.736
    >100, 60/468, d1=0.678, d2=0.697 g=0.767
    >100, 61/468, d1=0.611, d2=0.688 g=0.786
    >100, 62/468, d1=0.664, d2=0.654 g=0.796
    >100, 63/468, d1=0.663, d2=0.679 g=0.778
    >100, 64/468, d1=0.652, d2=0.666 g=0.763
    >100, 65/468, d1=0.683, d2=0.695 g=0.763
    >100, 66/468, d1=0.691, d2=0.656 g=0.776
    >100, 67/468, d1=0.643, d2=0.674 g=0.754
    >100, 68/468, d1=0.680, d2=0.641 g=0.750
    >100, 69/468, d1=0.712, d2=0.688 g=0.764
    >100, 70/468, d1=0.674, d2=0.733 g=0.789
    >100, 71/468, d1=0.697, d2=0.668 g=0.774
    >100, 72/468, d1=0.731, d2=0.680 g=0.779
    >100, 73/468, d1=0.713, d2=0.631 g=0.776
    >100, 74/468, d1=0.698, d2=0.635 g=0.780
    >100, 75/468, d1=0.687, d2=0.652 g=0.810
    >100, 76/468, d1=0.688, d2=0.679 g=0.798
    >100, 77/468, d1=0.684, d2=0.626 g=0.803
    >100, 78/468, d1=0.640, d2=0.600 g=0.795
    >100, 79/468, d1=0.701, d2=0.623 g=0.831
    >100, 80/468, d1=0.659, d2=0.623 g=0.798
    >100, 81/468, d1=0.692, d2=0.646 g=0.795
    >100, 82/468, d1=0.699, d2=0.625 g=0.813
    >100, 83/468, d1=0.684, d2=0.616 g=0.834
    >100, 84/468, d1=0.742, d2=0.587 g=0.828
    >100, 85/468, d1=0.711, d2=0.648 g=0.819
    >100, 86/468, d1=0.696, d2=0.617 g=0.809
    >100, 87/468, d1=0.667, d2=0.659 g=0.862
    >100, 88/468, d1=0.670, d2=0.645 g=0.794
    >100, 89/468, d1=0.722, d2=0.641 g=0.813
    >100, 90/468, d1=0.680, d2=0.660 g=0.794
    >100, 91/468, d1=0.703, d2=0.684 g=0.809
    >100, 92/468, d1=0.661, d2=0.668 g=0.787
    >100, 93/468, d1=0.643, d2=0.686 g=0.755
    >100, 94/468, d1=0.636, d2=0.648 g=0.769
    >100, 95/468, d1=0.671, d2=0.672 g=0.770
    >100, 96/468, d1=0.649, d2=0.694 g=0.724
    >100, 97/468, d1=0.636, d2=0.656 g=0.750
    >100, 98/468, d1=0.645, d2=0.652 g=0.758
    >100, 99/468, d1=0.634, d2=0.685 g=0.771
    >100, 100/468, d1=0.622, d2=0.661 g=0.760
    >100, 101/468, d1=0.635, d2=0.653 g=0.780
    >100, 102/468, d1=0.653, d2=0.686 g=0.744
    >100, 103/468, d1=0.613, d2=0.661 g=0.795
    >100, 104/468, d1=0.642, d2=0.672 g=0.764
    >100, 105/468, d1=0.655, d2=0.652 g=0.745
    >100, 106/468, d1=0.656, d2=0.702 g=0.766
    >100, 107/468, d1=0.646, d2=0.626 g=0.779
    >100, 108/468, d1=0.695, d2=0.691 g=0.746
    >100, 109/468, d1=0.664, d2=0.714 g=0.796
    >100, 110/468, d1=0.633, d2=0.656 g=0.768
    >100, 111/468, d1=0.645, d2=0.643 g=0.748
    >100, 112/468, d1=0.672, d2=0.651 g=0.789
    >100, 113/468, d1=0.677, d2=0.666 g=0.796
    >100, 114/468, d1=0.613, d2=0.678 g=0.770
    >100, 115/468, d1=0.643, d2=0.655 g=0.747
    >100, 116/468, d1=0.659, d2=0.684 g=0.761
    >100, 117/468, d1=0.696, d2=0.701 g=0.771
    >100, 118/468, d1=0.636, d2=0.690 g=0.748
    >100, 119/468, d1=0.661, d2=0.690 g=0.810
    >100, 120/468, d1=0.658, d2=0.664 g=0.791
    >100, 121/468, d1=0.579, d2=0.659 g=0.784
    >100, 122/468, d1=0.650, d2=0.670 g=0.782
    >100, 123/468, d1=0.605, d2=0.639 g=0.790
    >100, 124/468, d1=0.613, d2=0.652 g=0.769
    >100, 125/468, d1=0.682, d2=0.658 g=0.809
    >100, 126/468, d1=0.663, d2=0.604 g=0.825
    >100, 127/468, d1=0.680, d2=0.644 g=0.787
    >100, 128/468, d1=0.629, d2=0.675 g=0.805
    >100, 129/468, d1=0.650, d2=0.651 g=0.806
    >100, 130/468, d1=0.589, d2=0.655 g=0.814
    >100, 131/468, d1=0.696, d2=0.634 g=0.827
    >100, 132/468, d1=0.636, d2=0.668 g=0.823
    >100, 133/468, d1=0.648, d2=0.666 g=0.777
    >100, 134/468, d1=0.624, d2=0.620 g=0.824
    >100, 135/468, d1=0.663, d2=0.593 g=0.793
    >100, 136/468, d1=0.649, d2=0.609 g=0.786
    >100, 137/468, d1=0.674, d2=0.722 g=0.807
    >100, 138/468, d1=0.641, d2=0.623 g=0.811
    >100, 139/468, d1=0.622, d2=0.643 g=0.735
    >100, 140/468, d1=0.695, d2=0.658 g=0.746
    >100, 141/468, d1=0.661, d2=0.675 g=0.815
    >100, 142/468, d1=0.677, d2=0.609 g=0.810
    >100, 143/468, d1=0.683, d2=0.668 g=0.786
    >100, 144/468, d1=0.697, d2=0.679 g=0.799
    >100, 145/468, d1=0.677, d2=0.649 g=0.751
    >100, 146/468, d1=0.687, d2=0.677 g=0.757
    >100, 147/468, d1=0.673, d2=0.672 g=0.770
    >100, 148/468, d1=0.648, d2=0.632 g=0.803
    >100, 149/468, d1=0.712, d2=0.634 g=0.813
    >100, 150/468, d1=0.715, d2=0.645 g=0.806
    >100, 151/468, d1=0.741, d2=0.678 g=0.778
    >100, 152/468, d1=0.717, d2=0.641 g=0.768
    >100, 153/468, d1=0.659, d2=0.667 g=0.825
    >100, 154/468, d1=0.714, d2=0.666 g=0.817
    >100, 155/468, d1=0.757, d2=0.668 g=0.820
    >100, 156/468, d1=0.686, d2=0.614 g=0.809
    >100, 157/468, d1=0.691, d2=0.635 g=0.798
    >100, 158/468, d1=0.726, d2=0.701 g=0.793
    >100, 159/468, d1=0.687, d2=0.619 g=0.845
    >100, 160/468, d1=0.699, d2=0.658 g=0.759
    >100, 161/468, d1=0.694, d2=0.684 g=0.769
    >100, 162/468, d1=0.622, d2=0.703 g=0.793
    >100, 163/468, d1=0.703, d2=0.666 g=0.799
    >100, 164/468, d1=0.729, d2=0.668 g=0.762
    >100, 165/468, d1=0.664, d2=0.658 g=0.772
    >100, 166/468, d1=0.648, d2=0.637 g=0.730
    >100, 167/468, d1=0.639, d2=0.691 g=0.790
    >100, 168/468, d1=0.674, d2=0.678 g=0.785
    >100, 169/468, d1=0.713, d2=0.668 g=0.786
    >100, 170/468, d1=0.703, d2=0.677 g=0.782
    >100, 171/468, d1=0.706, d2=0.716 g=0.727
    >100, 172/468, d1=0.675, d2=0.697 g=0.720
    >100, 173/468, d1=0.659, d2=0.683 g=0.757
    >100, 174/468, d1=0.644, d2=0.729 g=0.725
    >100, 175/468, d1=0.664, d2=0.721 g=0.710
    >100, 176/468, d1=0.620, d2=0.724 g=0.732
    >100, 177/468, d1=0.633, d2=0.746 g=0.738
    >100, 178/468, d1=0.685, d2=0.755 g=0.730
    >100, 179/468, d1=0.664, d2=0.701 g=0.733
    >100, 180/468, d1=0.725, d2=0.678 g=0.728
    >100, 181/468, d1=0.676, d2=0.658 g=0.727
    >100, 182/468, d1=0.656, d2=0.696 g=0.711
    >100, 183/468, d1=0.619, d2=0.681 g=0.741
    >100, 184/468, d1=0.655, d2=0.736 g=0.708
    >100, 185/468, d1=0.660, d2=0.718 g=0.739
    >100, 186/468, d1=0.649, d2=0.691 g=0.724
    >100, 187/468, d1=0.731, d2=0.714 g=0.760
    >100, 188/468, d1=0.652, d2=0.696 g=0.739
    >100, 189/468, d1=0.664, d2=0.609 g=0.754
    >100, 190/468, d1=0.645, d2=0.724 g=0.746
    >100, 191/468, d1=0.632, d2=0.709 g=0.740
    >100, 192/468, d1=0.666, d2=0.749 g=0.754
    >100, 193/468, d1=0.695, d2=0.647 g=0.766
    >100, 194/468, d1=0.672, d2=0.635 g=0.819
    >100, 195/468, d1=0.650, d2=0.681 g=0.778
    >100, 196/468, d1=0.650, d2=0.680 g=0.776
    >100, 197/468, d1=0.612, d2=0.675 g=0.772
    >100, 198/468, d1=0.657, d2=0.681 g=0.774
    >100, 199/468, d1=0.601, d2=0.691 g=0.781
    >100, 200/468, d1=0.676, d2=0.662 g=0.774
    >100, 201/468, d1=0.688, d2=0.683 g=0.818
    >100, 202/468, d1=0.703, d2=0.665 g=0.793
    >100, 203/468, d1=0.643, d2=0.717 g=0.808
    >100, 204/468, d1=0.610, d2=0.661 g=0.764
    >100, 205/468, d1=0.613, d2=0.649 g=0.810
    >100, 206/468, d1=0.695, d2=0.672 g=0.817
    >100, 207/468, d1=0.655, d2=0.649 g=0.812
    >100, 208/468, d1=0.673, d2=0.693 g=0.798
    >100, 209/468, d1=0.678, d2=0.646 g=0.804
    >100, 210/468, d1=0.680, d2=0.626 g=0.801
    >100, 211/468, d1=0.650, d2=0.657 g=0.856
    >100, 212/468, d1=0.692, d2=0.621 g=0.832
    >100, 213/468, d1=0.665, d2=0.659 g=0.822
    >100, 214/468, d1=0.670, d2=0.640 g=0.821
    >100, 215/468, d1=0.655, d2=0.636 g=0.804
    >100, 216/468, d1=0.654, d2=0.672 g=0.820
    >100, 217/468, d1=0.634, d2=0.668 g=0.848
    >100, 218/468, d1=0.703, d2=0.625 g=0.841
    >100, 219/468, d1=0.693, d2=0.596 g=0.815
    >100, 220/468, d1=0.696, d2=0.637 g=0.849
    >100, 221/468, d1=0.700, d2=0.648 g=0.881
    >100, 222/468, d1=0.701, d2=0.567 g=0.882
    >100, 223/468, d1=0.708, d2=0.593 g=0.873
    >100, 224/468, d1=0.672, d2=0.630 g=0.866
    >100, 225/468, d1=0.658, d2=0.606 g=0.912
    >100, 226/468, d1=0.698, d2=0.633 g=0.877
    >100, 227/468, d1=0.691, d2=0.608 g=0.887
    >100, 228/468, d1=0.677, d2=0.629 g=0.807
    >100, 229/468, d1=0.710, d2=0.549 g=0.853
    >100, 230/468, d1=0.658, d2=0.607 g=0.846
    >100, 231/468, d1=0.720, d2=0.593 g=0.833
    >100, 232/468, d1=0.684, d2=0.636 g=0.850
    >100, 233/468, d1=0.708, d2=0.660 g=0.821
    >100, 234/468, d1=0.638, d2=0.683 g=0.785
    >100, 235/468, d1=0.647, d2=0.649 g=0.805
    >100, 236/468, d1=0.608, d2=0.713 g=0.823
    >100, 237/468, d1=0.654, d2=0.662 g=0.800
    >100, 238/468, d1=0.614, d2=0.712 g=0.798
    >100, 239/468, d1=0.682, d2=0.699 g=0.811
    >100, 240/468, d1=0.673, d2=0.682 g=0.770
    >100, 241/468, d1=0.625, d2=0.744 g=0.819
    >100, 242/468, d1=0.664, d2=0.648 g=0.792
    >100, 243/468, d1=0.723, d2=0.647 g=0.747
    >100, 244/468, d1=0.654, d2=0.660 g=0.780
    >100, 245/468, d1=0.669, d2=0.686 g=0.758
    >100, 246/468, d1=0.664, d2=0.727 g=0.778
    >100, 247/468, d1=0.641, d2=0.735 g=0.745
    >100, 248/468, d1=0.618, d2=0.750 g=0.744
    >100, 249/468, d1=0.625, d2=0.697 g=0.745
    >100, 250/468, d1=0.653, d2=0.766 g=0.722
    >100, 251/468, d1=0.638, d2=0.699 g=0.762
    >100, 252/468, d1=0.663, d2=0.719 g=0.762
    >100, 253/468, d1=0.619, d2=0.691 g=0.712
    >100, 254/468, d1=0.619, d2=0.724 g=0.722
    >100, 255/468, d1=0.678, d2=0.740 g=0.744
    >100, 256/468, d1=0.672, d2=0.697 g=0.743
    >100, 257/468, d1=0.638, d2=0.703 g=0.743
    >100, 258/468, d1=0.664, d2=0.771 g=0.757
    >100, 259/468, d1=0.666, d2=0.683 g=0.804
    >100, 260/468, d1=0.684, d2=0.667 g=0.753
    >100, 261/468, d1=0.632, d2=0.683 g=0.816
    >100, 262/468, d1=0.683, d2=0.679 g=0.805
    >100, 263/468, d1=0.654, d2=0.649 g=0.787
    >100, 264/468, d1=0.689, d2=0.752 g=0.784
    >100, 265/468, d1=0.719, d2=0.707 g=0.773
    >100, 266/468, d1=0.685, d2=0.708 g=0.782
    >100, 267/468, d1=0.619, d2=0.645 g=0.776
    >100, 268/468, d1=0.667, d2=0.694 g=0.768
    >100, 269/468, d1=0.620, d2=0.693 g=0.769
    >100, 270/468, d1=0.618, d2=0.737 g=0.775
    >100, 271/468, d1=0.709, d2=0.716 g=0.797
    >100, 272/468, d1=0.656, d2=0.683 g=0.740
    >100, 273/468, d1=0.676, d2=0.676 g=0.779
    >100, 274/468, d1=0.670, d2=0.667 g=0.779
    >100, 275/468, d1=0.640, d2=0.642 g=0.831
    >100, 276/468, d1=0.692, d2=0.653 g=0.753
    >100, 277/468, d1=0.646, d2=0.685 g=0.856
    >100, 278/468, d1=0.628, d2=0.668 g=0.809
    >100, 279/468, d1=0.665, d2=0.685 g=0.811
    >100, 280/468, d1=0.681, d2=0.589 g=0.768
    >100, 281/468, d1=0.697, d2=0.696 g=0.790
    >100, 282/468, d1=0.670, d2=0.642 g=0.741
    >100, 283/468, d1=0.713, d2=0.678 g=0.762
    >100, 284/468, d1=0.725, d2=0.703 g=0.743
    >100, 285/468, d1=0.662, d2=0.642 g=0.768
    >100, 286/468, d1=0.748, d2=0.677 g=0.761
    >100, 287/468, d1=0.677, d2=0.700 g=0.723
    >100, 288/468, d1=0.675, d2=0.678 g=0.771
    >100, 289/468, d1=0.720, d2=0.661 g=0.781
    >100, 290/468, d1=0.699, d2=0.667 g=0.782
    >100, 291/468, d1=0.713, d2=0.686 g=0.823
    >100, 292/468, d1=0.724, d2=0.618 g=0.778
    >100, 293/468, d1=0.718, d2=0.615 g=0.798
    >100, 294/468, d1=0.690, d2=0.623 g=0.812
    >100, 295/468, d1=0.699, d2=0.642 g=0.830
    >100, 296/468, d1=0.738, d2=0.606 g=0.823
    >100, 297/468, d1=0.735, d2=0.589 g=0.850
    >100, 298/468, d1=0.671, d2=0.624 g=0.839
    >100, 299/468, d1=0.757, d2=0.614 g=0.854
    >100, 300/468, d1=0.736, d2=0.554 g=0.864
    >100, 301/468, d1=0.670, d2=0.596 g=0.868
    >100, 302/468, d1=0.666, d2=0.602 g=0.858
    >100, 303/468, d1=0.728, d2=0.643 g=0.881
    >100, 304/468, d1=0.668, d2=0.593 g=0.861
    >100, 305/468, d1=0.712, d2=0.596 g=0.866
    >100, 306/468, d1=0.676, d2=0.654 g=0.863
    >100, 307/468, d1=0.706, d2=0.564 g=0.832
    >100, 308/468, d1=0.667, d2=0.708 g=0.885
    >100, 309/468, d1=0.617, d2=0.638 g=0.809
    >100, 310/468, d1=0.593, d2=0.579 g=0.868
    >100, 311/468, d1=0.663, d2=0.692 g=0.813
    >100, 312/468, d1=0.691, d2=0.687 g=0.822
    >100, 313/468, d1=0.602, d2=0.660 g=0.864
    >100, 314/468, d1=0.687, d2=0.675 g=0.787
    >100, 315/468, d1=0.651, d2=0.681 g=0.801
    >100, 316/468, d1=0.617, d2=0.665 g=0.788
    >100, 317/468, d1=0.653, d2=0.668 g=0.812
    >100, 318/468, d1=0.617, d2=0.666 g=0.805
    >100, 319/468, d1=0.657, d2=0.686 g=0.750
    >100, 320/468, d1=0.618, d2=0.750 g=0.732
    >100, 321/468, d1=0.684, d2=0.742 g=0.774
    >100, 322/468, d1=0.619, d2=0.685 g=0.777
    >100, 323/468, d1=0.653, d2=0.706 g=0.788
    >100, 324/468, d1=0.683, d2=0.740 g=0.720
    >100, 325/468, d1=0.671, d2=0.650 g=0.776
    >100, 326/468, d1=0.683, d2=0.709 g=0.747
    >100, 327/468, d1=0.650, d2=0.697 g=0.753
    >100, 328/468, d1=0.645, d2=0.689 g=0.741
    >100, 329/468, d1=0.649, d2=0.737 g=0.740
    >100, 330/468, d1=0.599, d2=0.675 g=0.744
    >100, 331/468, d1=0.628, d2=0.654 g=0.726
    >100, 332/468, d1=0.582, d2=0.697 g=0.731
    >100, 333/468, d1=0.583, d2=0.680 g=0.754
    >100, 334/468, d1=0.680, d2=0.688 g=0.784
    >100, 335/468, d1=0.619, d2=0.690 g=0.829
    >100, 336/468, d1=0.618, d2=0.673 g=0.783
    >100, 337/468, d1=0.674, d2=0.617 g=0.785
    >100, 338/468, d1=0.616, d2=0.664 g=0.805
    >100, 339/468, d1=0.599, d2=0.674 g=0.810
    >100, 340/468, d1=0.612, d2=0.647 g=0.790
    >100, 341/468, d1=0.591, d2=0.644 g=0.822
    >100, 342/468, d1=0.620, d2=0.661 g=0.816
    >100, 343/468, d1=0.664, d2=0.611 g=0.838
    >100, 344/468, d1=0.667, d2=0.696 g=0.833
    >100, 345/468, d1=0.603, d2=0.617 g=0.830
    >100, 346/468, d1=0.664, d2=0.655 g=0.796
    >100, 347/468, d1=0.634, d2=0.617 g=0.797
    >100, 348/468, d1=0.654, d2=0.650 g=0.857
    >100, 349/468, d1=0.595, d2=0.647 g=0.806
    >100, 350/468, d1=0.633, d2=0.678 g=0.822
    >100, 351/468, d1=0.650, d2=0.656 g=0.780
    >100, 352/468, d1=0.670, d2=0.644 g=0.755
    >100, 353/468, d1=0.627, d2=0.720 g=0.777
    >100, 354/468, d1=0.694, d2=0.754 g=0.757
    >100, 355/468, d1=0.653, d2=0.694 g=0.760
    >100, 356/468, d1=0.673, d2=0.720 g=0.779
    >100, 357/468, d1=0.753, d2=0.706 g=0.782
    >100, 358/468, d1=0.705, d2=0.642 g=0.778
    >100, 359/468, d1=0.769, d2=0.665 g=0.784
    >100, 360/468, d1=0.747, d2=0.653 g=0.799
    >100, 361/468, d1=0.702, d2=0.595 g=0.833
    >100, 362/468, d1=0.722, d2=0.663 g=0.865
    >100, 363/468, d1=0.757, d2=0.595 g=0.855
    >100, 364/468, d1=0.758, d2=0.583 g=0.821
    >100, 365/468, d1=0.759, d2=0.635 g=0.884
    >100, 366/468, d1=0.801, d2=0.598 g=0.889
    >100, 367/468, d1=0.802, d2=0.558 g=0.882
    >100, 368/468, d1=0.735, d2=0.571 g=0.912
    >100, 369/468, d1=0.748, d2=0.582 g=0.865
    >100, 370/468, d1=0.742, d2=0.549 g=0.871
    >100, 371/468, d1=0.682, d2=0.566 g=0.896
    >100, 372/468, d1=0.691, d2=0.583 g=0.836
    >100, 373/468, d1=0.695, d2=0.611 g=0.880
    >100, 374/468, d1=0.706, d2=0.581 g=0.894
    >100, 375/468, d1=0.721, d2=0.544 g=0.858
    >100, 376/468, d1=0.668, d2=0.594 g=0.885
    >100, 377/468, d1=0.665, d2=0.593 g=0.901
    >100, 378/468, d1=0.644, d2=0.638 g=0.917
    >100, 379/468, d1=0.702, d2=0.622 g=0.894
    >100, 380/468, d1=0.645, d2=0.549 g=0.941
    >100, 381/468, d1=0.670, d2=0.684 g=0.972
    >100, 382/468, d1=0.682, d2=0.646 g=0.896
    >100, 383/468, d1=0.673, d2=0.585 g=0.882
    >100, 384/468, d1=0.777, d2=0.644 g=0.884
    >100, 385/468, d1=0.604, d2=0.624 g=0.818
    >100, 386/468, d1=0.634, d2=0.632 g=0.843
    >100, 387/468, d1=0.707, d2=0.597 g=0.821
    >100, 388/468, d1=0.643, d2=0.667 g=0.814
    >100, 389/468, d1=0.656, d2=0.696 g=0.769
    >100, 390/468, d1=0.703, d2=0.690 g=0.792
    >100, 391/468, d1=0.577, d2=0.671 g=0.785
    >100, 392/468, d1=0.559, d2=0.630 g=0.740
    >100, 393/468, d1=0.663, d2=0.698 g=0.725
    >100, 394/468, d1=0.701, d2=0.727 g=0.726
    >100, 395/468, d1=0.651, d2=0.743 g=0.696
    >100, 396/468, d1=0.669, d2=0.748 g=0.681
    >100, 397/468, d1=0.693, d2=0.720 g=0.704
    >100, 398/468, d1=0.627, d2=0.790 g=0.684
    >100, 399/468, d1=0.615, d2=0.763 g=0.673
    >100, 400/468, d1=0.706, d2=0.778 g=0.652
    >100, 401/468, d1=0.602, d2=0.773 g=0.672
    >100, 402/468, d1=0.648, d2=0.794 g=0.666
    >100, 403/468, d1=0.583, d2=0.750 g=0.672
    >100, 404/468, d1=0.598, d2=0.729 g=0.693
    >100, 405/468, d1=0.607, d2=0.709 g=0.691
    >100, 406/468, d1=0.641, d2=0.670 g=0.722
    >100, 407/468, d1=0.631, d2=0.705 g=0.757
    >100, 408/468, d1=0.628, d2=0.715 g=0.761
    >100, 409/468, d1=0.666, d2=0.680 g=0.781
    >100, 410/468, d1=0.649, d2=0.683 g=0.796
    >100, 411/468, d1=0.632, d2=0.642 g=0.794
    >100, 412/468, d1=0.620, d2=0.625 g=0.800
    >100, 413/468, d1=0.654, d2=0.623 g=0.834
    >100, 414/468, d1=0.598, d2=0.574 g=0.840
    >100, 415/468, d1=0.555, d2=0.625 g=0.869
    >100, 416/468, d1=0.606, d2=0.597 g=0.839
    >100, 417/468, d1=0.600, d2=0.670 g=0.910
    >100, 418/468, d1=0.640, d2=0.646 g=0.894
    >100, 419/468, d1=0.582, d2=0.619 g=0.834
    >100, 420/468, d1=0.601, d2=0.614 g=0.918
    >100, 421/468, d1=0.555, d2=0.615 g=0.899
    >100, 422/468, d1=0.594, d2=0.634 g=0.874
    >100, 423/468, d1=0.590, d2=0.627 g=0.898
    >100, 424/468, d1=0.546, d2=0.677 g=0.846
    >100, 425/468, d1=0.592, d2=0.645 g=0.834
    >100, 426/468, d1=0.690, d2=0.719 g=0.892
    >100, 427/468, d1=0.598, d2=0.687 g=0.925
    >100, 428/468, d1=0.605, d2=0.679 g=0.835
    >100, 429/468, d1=0.578, d2=0.661 g=0.820
    >100, 430/468, d1=0.627, d2=0.657 g=0.839
    >100, 431/468, d1=0.674, d2=0.625 g=0.864
    >100, 432/468, d1=0.697, d2=0.674 g=0.884
    >100, 433/468, d1=0.686, d2=0.680 g=0.839
    >100, 434/468, d1=0.644, d2=0.639 g=0.826
    >100, 435/468, d1=0.693, d2=0.600 g=0.884
    >100, 436/468, d1=0.683, d2=0.692 g=0.845
    >100, 437/468, d1=0.697, d2=0.655 g=0.830
    >100, 438/468, d1=0.751, d2=0.590 g=0.857
    >100, 439/468, d1=0.749, d2=0.663 g=0.817
    >100, 440/468, d1=0.728, d2=0.603 g=0.831
    >100, 441/468, d1=0.755, d2=0.634 g=0.857
    >100, 442/468, d1=0.726, d2=0.614 g=0.845
    >100, 443/468, d1=0.706, d2=0.618 g=0.845
    >100, 444/468, d1=0.800, d2=0.614 g=0.890
    >100, 445/468, d1=0.729, d2=0.597 g=0.890
    >100, 446/468, d1=0.721, d2=0.611 g=0.868
    >100, 447/468, d1=0.726, d2=0.568 g=0.869
    >100, 448/468, d1=0.733, d2=0.602 g=0.842
    >100, 449/468, d1=0.715, d2=0.577 g=0.884
    >100, 450/468, d1=0.693, d2=0.563 g=0.890
    >100, 451/468, d1=0.743, d2=0.580 g=0.903
    >100, 452/468, d1=0.767, d2=0.563 g=0.917
    >100, 453/468, d1=0.692, d2=0.554 g=0.903
    >100, 454/468, d1=0.709, d2=0.553 g=0.913
    >100, 455/468, d1=0.671, d2=0.588 g=0.917
    >100, 456/468, d1=0.663, d2=0.581 g=0.950
    >100, 457/468, d1=0.732, d2=0.569 g=0.916
    >100, 458/468, d1=0.736, d2=0.532 g=0.960
    >100, 459/468, d1=0.645, d2=0.595 g=0.975
    >100, 460/468, d1=0.626, d2=0.560 g=0.952
    >100, 461/468, d1=0.753, d2=0.519 g=0.897
    >100, 462/468, d1=0.642, d2=0.595 g=0.912
    >100, 463/468, d1=0.658, d2=0.567 g=0.899
    >100, 464/468, d1=0.747, d2=0.615 g=0.889
    >100, 465/468, d1=0.585, d2=0.650 g=0.840
    >100, 466/468, d1=0.652, d2=0.685 g=0.898
    >100, 467/468, d1=0.645, d2=0.566 g=0.870
    >100, 468/468, d1=0.656, d2=0.677 g=0.790
    

### Generate 16 random handwritten digits


```python
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()
 
# load model
model = load_model('generator.h5')
# generate images
latent_points = generate_latent_points(16, 16)
# generate images
X = model.predict(latent_points)
# plot the result
show_plot(X, 4)
```

WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
    


    
![svg](/static/post/Impelentation_of_GAN_and_cGAN/output_35_1.svg)
    


As conclution between two aproach above, we can see that there is no significant difference on both models. Event the result of the cGAN model looks a bit good than the GAN model, but it's no much different. But when we go to training time, honestly the cGAN model is quite fast than the GAN model. Unfortunately I did record time every model at that moment, so I can not explain how fast the cGAN model is compared to the GAN model.
