"""Validation exercises for the multilayer perceptron: not submitted, but they catch implementation bugs."""

# TODO: one forward and backward step of [2, 2, 1] matches your hand calculation
# TODO: one forward and backward step of [2, 3, 2, 1] matches your hand calculation
# TODO: learns XOR: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [1, 1, -1, -1]
# test 

from src.perceptrons.metrics import confusion_matrix, accuracy
from src.perceptrons.metrics import plot_confusion_matrix
from src.perceptrons.multilayer import multilayer, forward_propagation, one_hot, one_hot_decode
from src.perceptrons.data import X_valid, Y_valid

# Temporarily random seed: 
random_seed = np.random.default_rng(42)

def test_function(beta, function, layers, X_eval, Y_eval, W, b, name="Validation"):
    _, _, O = forward_propagation(beta, function, layers, X_eval, W, b)
    acc = accuracy(O, one_hot(Y_eval))
    print("{} accuracy: {:.3f}".format(name, acc))
    confusion_matrix(Y_eval, one_hot_decode(O))
    plot_confusion_matrix(Y_eval, one_hot_decode(O), 10)

beta = 1
eta = 0.5
W, b = multilayer(eta, beta, "sigmoid", [64], 2000, random_seed)
# Tune on the validation split; digits_test.csv is only for the final result
test_function(beta, "sigmoid", [64], X_valid, Y_valid, W, b)