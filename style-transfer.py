#!/usr/bin/env python3
"""
Image Stylization Using Deep Learning

Implemented by:
Regina Pramesti (u5533512)
Michelle Adiwangsa (u5888336)
Muhammad Khan (u6017002)

References:
    - https://arxiv.org/pdf/1508.06576.pdf (The research paper that this implementation is based on)
    - http://math.uga.edu/~rothstei/6120Spring2008/GramMatrix20080325.pdf (Gram Matrix mathematical formula)
    - https://towardsdatascience.com/neural-style-transfer-tutorial-part-1-f5cd3315fa7f (Referred to some parts of this tutorial for the gradient descent and loading VGG19)
"""

# Tensorflow for gradient descent
import tensorflow as tf

# Keras libraries for the tensors calculation
import keras
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model

# Image library for reading from and writing to images
from PIL import Image 

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Image stylization using a CNN')
parser.add_argument('content_image', metavar='content', type=str, help='Path to the input content image')
parser.add_argument('style_image', metavar='style', type=str, help='Path to the input style image')
parser.add_argument('result_image_prefix', metavar='res_prefix', type=str, help='Path to the output generated image')
parser.add_argument('--iterations', type=int, default=200, required=False, help='Number of iterations')
parser.add_argument('--style_weight', type=float, default=0.9, required=False, help='Style weight')
parser.add_argument('--content_weight', type=float, default=0.1, required=False, help='Content weight')

args = parser.parse_args()

iterations = args.iterations

####################
# Global Variables
####################

# Image parameters
num_channels = 3
height = 512
width = 512
image_size = height * width

# Path to images
content_path = args.content_image
style_path = args.style_image
target_path = args.result_image_prefix

# Layers in the CNN to be used for style and content
content_layers = ['block2_conv2']
style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
# These are the style layers used in the original paper
# style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

content_layers_count = len(content_layers)
style_layers_count = len(style_layers) 

# Loads and preprocesses images
def process_images(path):
    # Load the image
    image = Image.open(path)

    # Resize the image so that both content 
    # and style images are the same height
    image = image.resize((width, height))

    # Convert the image to a tensor of image data
    image = np.asarray(image, dtype='float32')

    # Add another dimension to the image tensor
    image = np.expand_dims(image, axis=0)

    # Ensure that only the first three channels are used
    image = image[:, :, :, :3]

    # Subtract the mean for preprocessing
    # Calculated Means: R:103.939, G:116.779, B:123.68
    image[:, :, :, 0] -= 103.939
    image[:, :, :, 1] -= 116.779
    image[:, :, :, 2] -= 123.68

    # Flip the channels from RGB to BGR, based on the paper
    image = image[:, :, :, ::-1]

    return image

# Builds a model based on the intermediate layers of the CNN used for style and content
def get_model(content_layers, style_layers):
    # Load the CNN, pre-trained on imagenet
    vgg19 = VGG19(weights='imagenet', include_top=False)
    vgg19.trainable = False

    # Take the output of the layers based on the layers used
    style_model_outputs   =  [vgg19.get_layer(name).output for name in style_layers]
    content_model_outputs =  [vgg19.get_layer(name).output for name in content_layers]
    
    model_outputs = content_model_outputs + style_model_outputs

    # Build model 
    return Model(inputs = vgg19.input, outputs = model_outputs),  vgg19
    
# Calculates the content loss based on the formula provided in the paper
def compute_content_loss(content_features, generated_features):
    return K.sum(K.pow(content_features - generated_features, 2)) / 2
    
# Calculates the Gram Matrix based on the Gram Matrix formula
def gram_matrix(features):

    # Reshaping (reducing the dimension of the array) uses less memory
    # and speeds up the calculations
    channels = int(features.shape[-1])
    a = tf.reshape(features, [-1, channels])

    return K.dot(K.transpose(a), a)

# Calculates the style loss based on the formula provided in the paper
def compute_style_loss(style_matrix, generated_matrix):
    # Get the gram matrices 
    style_mat = gram_matrix(style_matrix)
    generated_mat = gram_matrix(generated_matrix)

    # Calculate the style loss based on the equation in the paper
    return K.sum(K.pow(style_mat - generated_mat, 2)) / (4 * (num_channels ** 2) * (image_size ** 2))
    
# Calculates the total loss based on the formula provided in the paper
def total_loss(model, content_features, style_features, output_activations):

    content_activations = output_activations[:content_layers_count]
    style_activations = output_activations[content_layers_count:]

    # Use the weights that was passed as arguments when running this script
    content_weight = args.content_weight
    style_weight = args.style_weight

    content_loss = 0
    style_loss = 0

    # Iterate through all the content layers
    for target_content, comb_content in zip(content_features, content_activations):
        content_loss += compute_content_loss(comb_content[0], target_content) * (1.0 / content_layers_count)

    # Iterate through all the style layers
    for target_style, comb_style in zip(style_features, style_activations):
        style_loss += compute_style_loss(comb_style[0], target_style) * (1.0 / style_layers_count)

    # return content_weight * content_loss + style_weight * style_loss
    return content_weight * content_loss + style_weight * style_loss, style_loss, content_loss

# Write the contents of the generated image to an image file
def write_image(filename, generated):
    # Flip the channels back to RGB
    generated = generated.reshape((height, width, 3))
    generated = generated[:, :, ::-1]

    # Add the calculated means back
    generated[:, :, 0] += 103.939
    generated[:, :, 1] += 116.779
    generated[:, :, 2] += 123.68

    # Cast the values in the array to uint8
    generated = np.clip(generated, 0, 255).astype('uint8')

    # Save the image as the desired output filename
    output_image = Image.fromarray(generated)
    output_image.save(filename)

if __name__ == '__main__':

    # Create a tensorflow session 
    sess = tf.Session()

    # Assign keras back-end to the TF session which we created
    K.set_session(sess)

    # Get the Model object which includes all the layers requires
    # to compute the style and content losses
    model, vgg19 = get_model(content_layers,style_layers)
    
    # Create a variable to store the generated image
    # Instead of assigning random values on the first generated image, we
    # use the actual content image as it is seen to speed up convergence time
    generated = process_images(content_path)

    # Load and preprocess the content and style images
    content = process_images(content_path)
    style = process_images(style_path)

    # Converts the numpy arrays into Keras tensors
    content_image = K.variable(content)
    style_image = K.variable(style)
    generated_image = K.variable(generated)

    # Get the content and style feature representations 
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    generated_outputs = model(generated_image)

    # Features obtained based on the style and content layers specified
    # Use the gram matrix for style layers
    content_features = [ content_layer[0] for content_layer in content_outputs[:content_layers_count] ]
    style_features = [ style_layer[0]  for style_layer    in style_outputs[content_layers_count:] ]
    
    # Define loss and gradient
    loss = total_loss(model, content_features, style_features, generated_outputs)
    opt = tf.train.AdamOptimizer(learning_rate=9, beta1=0.9, epsilon=1e-1).minimize( loss[0], var_list = [generated_image])

    ## Initialise variables 
    sess.run(tf.global_variables_initializer())
    sess.run(generated_image.initializer)

    # Reload the weights of vgg19 because global_variables_initializer resets the weights
    vgg19.load_weights("vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")

    # Initialise the current best
    # Infinity so the coming iteration will be considered best loss and image
    best_loss, best_image = float('inf'), None

     # VGG default normalization
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means 

    for i in range(iterations):
        # Run the function that calculates the gradient for gradient descent
        sess.run(opt)

        # Make sure image values stays in the range of max-min value of VGG norm 
        clipped = tf.clip_by_value(generated_image, min_vals, max_vals)
        # assign the clipped value to the tensor stylized image
        generated_image.assign(clipped)

        # Open the Tuple of tensors 
        total_loss, style_score, content_score = loss
        total_loss = total_loss.eval(session=sess)

        if total_loss < best_loss:

            # Update best loss and best image from total loss
            best_loss = total_loss
            best_image = sess.run(generated_image)[0]
            
            s_loss = sess.run(style_score)
            c_loss = sess.run(content_score)

            # print best loss
            print('Iteration: ', i ,' Total loss: ', total_loss ,'  Style Loss: ',  s_loss,', Content_loss: ', c_loss)

        # Save the generated to a file
        output_filename = '{}-{}.png'.format(target_path, i+1)
        write_image(output_filename, best_image)

    # Close tensorflow session at the end
    sess.close()
