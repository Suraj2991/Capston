__author__ = 'suraj2991'
import theano as th
import numpy as np
import pdb
T = th.tensor
# foo = T.scalar('foo')
#
# bar = foo**2
# print type(bar)
# print bar.type
# print th.pp(bar)
#
# f = th.function([foo], bar)
# print f(3)
#
# print bar.eval({foo:3})
#
# A = T.matrix('A')
# x = T.vector('x')
# b = T.vector('b')
#
# y = T.dot(A,x) + b
# z = T.sum(A**2)
# linear_mix = th.function([A,x, th.Param(b, default=np.array([0,0]))], [y, z])
# print linear_mix(np.array([[1,2,3],[3,4,5]]), np.array([2,5,8]))
# print linear_mix(np.array([[1,2,3],[3,4,5]]), np.array([2,5,8]), np.array([2,3]))
# shared_var = th.shared(np.array([[2,3], [1,2]], dtype = th.config.floatX))
# shared_var.set_value(np.array([[1,2], [2,3]], dtype = th.config.floatX))
# print shared_var.get_value()
# shared_sq = shared_var ** 2
# func = th.function([], shared_sq)
# print func()
#
#
#
# subtract = T.matrix('subtract')
#
# func2 = th.function([subtract], shared_var, updates = {shared_var: shared_var - subtract})
#
# func2(np.array([[1,1], [1,1]]))
#
# print shared_var.get_value()
# print func()
#
# bar_grad = th.grad(bar, foo)
# print bar_grad.eval({foo:10})
#
# y_J = th.gradient.jacobian(y, x)
#
# linear_mix_J = th.function([A,x,b], y_J)
# print linear_mix_J(np.array([[9,8,7],[4,5,6]]), np.array([1,2,3]), np.array([4,5]))
#

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        :parameters:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise activation function for output layer
        '''
        print W_init.shape
        n_input, n_output = W_init.shape

        # assert b_init == (n_output)




        self.W = th.shared(value = W_init.astype(th.config.floatX), name = 'W', borrow = True)
        self.b = th.shared(value = b_init.reshape(-1,1).astype(th.config.floatX), name = 'b', borrow = True, broadcastable = (False,True ))
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, x):
        '''
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input
        '''
        lin_output = T.dot(self.W,x) + self.b
        if self.activation == None:
            return lin_output
        else:
            return self.activation(lin_output)

class MLP(object):
    def __init__(self, W_init, b_init, activations):
        '''
        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''

        assert len(W_init) == len(b_init) == len(activations)
        self.layers = []

        for W, b, activation in zip(W_init, b_init, activations):
            print W, b
            self.layers.append(Layer(W, b, activation))
        self.params = []
        for layer in self.layers:
            self.params += layer.params


    def output(self, x):
        '''
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
        '''
        for layer in self.layers:
            x = layer.output(x)
        return x


    def sqrd_error(self, x, y):
        '''
        :parameters:
            -x : theano.tensor.var.TensorVariable
                 network input
            -y : theano.tensor.var.TensorVariable
                desired network output
        '''
        return T.sum((self.output(x) - y)**2)


def gradient_updates(cost, params, learning_rate, momentum):
    '''

    :param cost: function to minimize
    :param params: parametrs to compute gradient
    :param learning_rate: gradient descent learning rate
    :param momentum: 0- standard gradient descent
                     <1 - momentum uses current gradient and previously computed gradient
    :return: updates of the parameters
    '''

    assert momentum <1 and momentum >=0

    updates = []
    for parameters in params:
        param_update = th.shared(value = parameters.get_value()*0., broadcastable = parameters.broadcastable)
        updates.append((parameters, parameters - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (1-momentum)* T.grad(cost, parameters)))
    return updates





def main():
    X= np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    N = 20
    Y = np.random.random_integers(0, 1, N)
    for i in range(20):
        X = np.vstack(np.random.random_integers(0,1,N))
    W = np.array([1.3, 2.4, 5.3, 1.1, 6.3, 1.1, 2.3, 2.2, 1.1, 2.1]).reshape(1,10)
    X = X.reshape(len(X),1)
    print (X)
    # fan input, layers, fan output
    layer_sizes = [X.shape[0],1, 1]
    W_init = []
    b_init = []
    activations = []
    print np.shape(W)
    assert np.shape(W) == (1, 10)
    n_inputs, n_outputs  = zip(layer_sizes[:-1], layer_sizes[1:])
    W_init.append(W)
    b_init.append(np.ones(n_outputs))
    activations.append(T.nnet.sigmoid)
    print np.shape(b_init)

    mlp = MLP(W_init, b_init, activations)
    mlp_input = T.matrix('mlp_input')
    mlp_target = T.vector('mlp_target')
    learning_rate = 0.01
    momentum = 0.9
    cost = mlp.sqrd_error(mlp_input, mlp_target)
    i = 1
    while i ==1:
        train = th.function([mlp_input, mlp_target], cost, updates= gradient_updates(cost, mlp.params, learning_rate, momentum))
        mlp_output = th.function([mlp_input], mlp.output(mlp_input))
        current_cost = train(X, Y)
        current_output = mlp_output(X)
        i = i +1
    print current_output, current_cost


main()