from keras.layers import Layer, Dense, InputSpec, Lambda, Input
from keras import activations
from keras import backend as K
from keras import initializations
from keras import regularizers
from keras import constraints
import numpy as np
import Queue
import warnings
import sys
import datetime
sys.setrecursionlimit(10000000)

def zeros(n):
    if K._BACKEND == 'theano':
        import theano.tensor as T
        return T.zeros(n)
    elif K._BACKEND == 'tensorflow':
        import tensorflow as tf
        return tf.zeros(n)


class HuffmanNode(object):
    def __init__(self, node_id = -1, left=None, right=None):
        self.left = left
        self.right = right
        self.node_id = node_id

    def children(self):
        return((self.left, self.right))


class Huffmax(Layer):
    '''
    inputs : [2D vector; float (batch_size, input_dim), 2D target classes; int (batch_size, nb_required_classes)]
    output: [2D probabilities; float (batch_size, nb_required_classes)]
    '''
    def __init__(self, nb_classes, frequencies = None, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, verbose=False, **kwargs):
        '''
        # Arguments:
        nb_classes: Number of classes.
        frequencies: list. Frequency of each class. More frequent classes will have shorter huffman codes.
        mode: integer. One of [0, 1]
        verbose: boolean. Set to true to see the progress of building huffman tree.
        '''
        self.nb_classes = nb_classes
        if not frequencies:
                frequencies = [(1, i)  for i in range(nb_classes)]
        self.frequencies = frequencies
        self.init = initializations.get(init)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.initial_weights = weights
        self.verbose = verbose
        self.huffman_codes = []
        # Generate leaves of Huffman tree.
        self.leaves = [Lambda(lambda x: K.cast(x * 0 + i, dtype='int32')) for i in range(self.nb_classes)]
        super(Huffmax, self).__init__(**kwargs)

    def build(self, input_shape):
        def create_huffman_tree(frequencies):
            p = Queue.PriorityQueue()
            for value in frequencies:
                p.put(value)

            node_id = 0
            while p.qsize() > 1:
                l, r = p.get(), p.get()
                node = (l[0]+r[0], HuffmanNode(node_id, l, r))
                node_id += 1
                p.put(node)
            return p.get()

        def assign_codes(node, code_prefix=[], path_prefix = [], paths = {}, code={}):
            if isinstance(node[1].left[1], HuffmanNode):
                assign_codes(node[1].left,code_prefix+[0], path_prefix + [node[1].node_id], paths, code)
            else:
                code[node[1].left[1]] = code_prefix+[0]
                paths[node[1].left[1]] = path_prefix+ [node[1].node_id]
            if isinstance(node[1].right[1],HuffmanNode):
                assign_codes(node[1].right,code_prefix+[1], path_prefix + [node[1].node_id], paths, code)
            else:
                code[node[1].right[1]] = code_prefix+[1]
                paths[node[1].right[1]] = path_prefix+[node[1].node_id]

            return code, paths

        if self.verbose:
            print('Build started')
        if type(input_shape) == list:
            self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=(input_shape[1]))]
        else:
            self.input_spec = [InputSpec(shape=input_shape)]
            input_shape = [input_shape, None]
        input_dim = input_shape[0][1]

        # Build Huffman tree.
        if self.verbose:
            print('Building huffman tree...')

        self.root_node = create_huffman_tree(self.frequencies)
        code_map, paths_map = assign_codes(self.root_node)

        max_tree_depth = max(map(len, code_map.values()))
        self.huffman_codes = [x[1] for x in sorted(code_map.items())]
        for huffman_code in self.huffman_codes:
            huffman_code += [0] * (max_tree_depth - len(huffman_code))
        self.huffman_codes = K.variable(self.huffman_codes)

        self.class_paths = [x[1] for x in sorted(paths_map.items())]
        for paths in self.class_paths:
            paths += [self.root_node[1].node_id] * (max_tree_depth - len(paths))
        self.class_paths = K.variable(self.class_paths, dtype='int32')

        total_nodes = len(code_map.keys())

        if self.verbose:
            print('Huffman tree build complete')
            print('Setting weights...')

        self.W = self.init((total_nodes, input_dim, 1))
        if self.bias:
            self.b = K.zeros((total_nodes, 1))
            self.trainable_weights = [self.W, self.b]
        else:

            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if hasattr(self, 'initial_weights') and self.initial_weights:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(Huffmax, self).build(input_shape)
        if self.verbose:
            print('Done.')


    def call(self, x, mask=None):
        input_vector = x[0]
        target_classes = x[1]
        nb_req_classes = self.input_spec[1].shape[1]
        if nb_req_classes is None:
            nb_req_classes = K.shape(target_classes)
        if K.dtype(target_classes) != 'int32':
            target_classes = K.cast(target_classes, 'int32')
        # One giant matrix mul
        input_dim = self.input_spec[0].shape[1]
        req_nodes = K.gather(self.class_paths, target_classes)
        req_W = K.gather(self.W, req_nodes)
        y = K.batch_dot(input_vector, req_W, axes=(1, 3))
        if self.bias:
            req_b = K.gather(self.b, req_nodes)
            y += req_b
        y = K.sigmoid(y[:, :, :, 0])
        req_huffman_codes = K.gather(huffman_codes, target_classes)
        return K.prod(req_huffman_codes + y - 2 * req_huffman_codes * y, axis=-1)  # Thug life

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[1][1])

    def get_config(self):
        config = {'nb_classes': self.nb_classes,
                  'frequency_table': self.frequencies,
                  'kwargs': self.kwargs
                  }
        base_config = super(Huffmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HuffmaxClassifier(Huffmax):
    ''' This layer is not differentiable. Hence, can be used for prediction only.
    Train the weights using the Huffmax layer, and transfer them here for prediction.
    For a given 2D input (batch_size, input_dim), outputs a 1D integer array of class labels.
    '''

    def __init__(self, nb_classes, input_dim, **kwargs):
        kwargs['nb_classes'] = nb_classes
        kwargs['input_shape'] = (input_dim,)
        super(HuffmaxClassifier, self).__init__(**kwargs)

    def call(self, x, mask=None):

        def get_node_w(node):
            return self.W[self.node_indices[node], :, :]

        def get_node_b(node):
            return self.b[self.node_indices[node], :]

        def compute_output(input, node=self.root_node):
            if not hasattr(node, 'left'):
                return zeros((K.shape(input)[0],)) + self.node_indices[node]
            else:
                node_output = K.dot(x, get_node_w(node))
                if self.bias:
                    node_output += get_node_b(node)
                left_prob = node_output[:, 0]
                right_prob = 1 - node_output[:, 0]
                left_node_output = compute_output(input, node.left)
                right_node_output = compute_output(input, node.right)
                return K.switch(left_prob > right_prob, left_node_output, right_node_output)
        return K.cast(compute_output(x), 'int32')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],)
