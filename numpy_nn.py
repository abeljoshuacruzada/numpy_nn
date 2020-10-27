import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def act_sigmoid(x, deriv=False):
    sigmoid = 1 / (1 + np.exp(-x))
    if deriv:
        return sigmoid * (1 - sigmoid)
    return sigmoid


def act_relu(x, deriv=False):
    if deriv:
        x[x > 0] = 1
        x[x <= 0] = 0
        return x
    else:
        return np.maximum(0, x)


def act_leaky_relu(x, alpha=0.01, deriv=False):
    if deriv:
        return (1 if x > 0 else alpha)
    else:
        return np.maximum(alpha * x, x)


def act_tanh(x, deriv=False):
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if deriv:
        return 1 - np.power(tanh, 2)
    return tanh


def act_sin(x, deriv=False):
    if deriv:
        return np.cos(x)
    else:
        return np.sin(x)


def create_regressor_nn(x, y, no_hidden_layer, no_hidden_nodes, gamma_list,
                        act_list, bias=False, seed=42):
    """
    Create a Regressor Neural Network implementation using numpy.
    Returns 3 tuple of (actual result, error, r2)

    Parameters
    ----------
    x : pandas Series
        x dataset
    y : pandas Series
        y output/target
    no_hidden_layer : int
        Number of hidden layer, excluding input and output layer
    no_hidden_nodes: array_like
        Array of number of nodes for each layer
    gamma_list : array_like
        Array of gamma for each layer
    act_list : array_like
        Array of activation functions
    bias : bool, default = False
        True to add a bias in the input layer otherwise False
    seed : int, defualt = 42
        Random seed to use

    Returns
    -------
        Returns 3 tuple of (actual result, error, r2)

    Examples
    --------
    >>> create_regressor_nn(X_train, y_train,
                 no_hidden_layer=2,
                 no_hidden_nodes=[3, 2],
                 gamma_list=[0.1, 0.01, 0.001],
                 act_list=[act_sigmoid, act_relu, act_relu],
                 bias=True)
    (array([[0.],
        [0.],
        [0.],
        [0.],
        ...
        [0.],
        [0.]]),
     4616.9726277372265,
     -5.8575129574207185)
    """
    if len(act_list) != len(gamma_list):
        print('Invalid number gamma in the gamma list')
        return

    if len(act_list) != (no_hidden_layer + 1):
        print('Invalid number of activation functions in the list')
        return

    x = x.copy()
    if bias:
        x.insert(0, 'bias', 1)
    x = x.to_numpy()
    y = y.to_numpy()[:, np.newaxis]
    no_inputs = x.shape[1]
    no_output = (1 if len(y.shape) == 1 else y.shape[1])
    np.random.seed(seed)

    # Initialize weights
    wlist = [2 * np.random.random((no_inputs, no_hidden_nodes[0])) - 1]
    for hidden in range(no_hidden_layer):
        if hidden == no_hidden_layer - 1:
            # Initial weights for last layer
            wlist.append(2 * np.random.random((no_hidden_nodes[hidden],
                                               no_output)) - 1)
        else:
            # Initial weights for middle layer
            wlist.append(2 * np.random.random((no_hidden_nodes[hidden],
                                               no_hidden_nodes[hidden + 1])) - 1)
    last_error = None
    i = 0
    while True:
        # Calculate new layers
        layers = [x]
        for hidden in range(no_hidden_layer + 1):
            layer = act_list[hidden](np.dot(layers[hidden],
                                            wlist[hidden]))
            layers.append(layer)

        # Check for error
        error = y - layers[-1]
        mean_error = np.mean(np.abs(error))
        if last_error is not None and last_error <= mean_error:
            print('Minimum error =', last_error)
            break
        else:
            last_error = mean_error

        if (i % 50000) == 0:
            print("Error =", np.mean(np.abs(mean_error)))

        # Gradients
        gradients = list()
        for ind in range(len(layers))[:0:-1]:
            # Loop from the last layer to the 2nd layer
            if ind == (len(layers) - 1):
                # Last layer
                # Layer, weight, activation function
                layer, w, act = layers[ind - 1], wlist[ind - 1], act_list[ind - 1]
                delta = (error * act(np.dot(layer, w), True))
            else:
                last_delta, w, act = (gradients[-1], wlist[ind],
                                      act_list[ind - 1])
                delta = (last_delta.dot(w.T) * act(np.dot(layers[ind - 1],
                                                          wlist[ind - 1]),
                                                   True))
            gradients.append(delta)

        # Adjust weights
        for w_ind in range(len(wlist)):
            w = wlist[w_ind]
            # Calculate new weight
            wlist[w_ind] = (w + gamma_list[w_ind] *
                            layers[w_ind].T.dot(gradients.pop()))
        i += 1
    actual = layers[-1]
    r2 = r2_score(y, actual)
    print('R2 =', r2)
    print('MSE =', mean_squared_error(y, actual))
    return actual, last_error, r2