# image-classification-caltech-256

### Summary
- explore_and_resize.ipynb
  - load and explore image metadata
  - resize images to 64*64
  - convert images to numpy arrays
  - save the arrays to a file (final size - 269 MB)

- basic_keras.ipynb
  - load images as a numpy array into memory
  - build an input pipeline
  - train a simple CNN with Keras
  - train accuracy: 0.2397, validation accuracy: 0.2723
  - training time: 15 min on p2.xlarge
  - number of parameters: 1,188,001
