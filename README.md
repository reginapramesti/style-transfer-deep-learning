# Image Stylization Using a Convolutional Neural Network

### Dependencies

* Python 3.6
* Keras
* TensorFlow
* NumPy

### Usage

`python3 style-transfer.py <content_image> <style_image> <output_filename> <optional_parameters>`

An example of this is:

`python3 style-transfer.py images/content/stata.jpg images/style/starry_night.jpg images/generated/starry_night_stata`

The entire run takes about 30 mins to 1 hour depending on the image, the weights of the content and style loss, and the style layers used in the VGG19 CNN.

### Optional Parameters

We have provided the option to run the Python script with different parameters, such as:

`--iterations <number of iterations>: this is the maximum number of iterations that the gradient descent will go through before the script ends`

`--style_weight <style weight value>: this is the weight associated with the style loss as part of the total loss function`

`--content_weight <content weight value>: this is the weight associated with the content loss as part of the total loss function`


