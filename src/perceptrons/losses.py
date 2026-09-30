"""Loss functions and their gradients with respect to the network output."""

# TODO: the loss for Exercise 1 (output is a probability)
# TODO: the loss for Exercises 2 and 3 (output is one of 10 classes)


def mse(Y_pred, Y_true):
    error = ((Y_pred - Y_true) ** 2).sum() / (2 * Y_pred.size) # Divide by batch_size?
    return error