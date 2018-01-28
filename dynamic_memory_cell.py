from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class DynamicMemoryCell(tf.contrib.rnn.RNNCell):
    """
    Implementation of a dynamic memory cell as a gated recurrent network.
    The cell's hidden state is divided into blocks and each block's weights are tied.
    """

    def __init__(self, num_blocks, num_units_per_block, keys, initializer=None, activation=tf.nn.relu):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._keys = keys
        self._activation = activation # \phi
        self._initializer = initializer

    @property
    def state_size(self):
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        #return self._num_blocks * self._num_units_per_block
        return self._num_blocks
    
    
    def zero_state(self, batch_size, dtype):
        """
        We initialize the memory to the key values.
        """
        zero_state = tf.concat([tf.expand_dims(key, 0) for key in self._keys], 1)
        zero_state_batch = tf.tile(zero_state, tf.stack([batch_size, 1]))
        return zero_state_batch
    
    '''
    def zero_state(self, batch_size, dtype): 
        zero_initial = [tf.constant(0.0, shape=[100]) for j in range(self._num_blocks)]
        zero_state = tf.concat([tf.expand_dims(item, 0) for item in zero_initial], 1)
        zero_state_batch = tf.tile(zero_state, tf.stack([batch_size, 1]))
        return zero_state_batch
    '''

    '''
    def get_gate(self, state_j, key_j, inputs):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)
        """
        a = tf.reduce_sum(inputs * state_j, reduction_indices=[1])
        b = tf.reduce_sum(inputs * tf.expand_dims(key_j, 0), reduction_indices=[1])
        return tf.sigmoid(a + b)
    '''

    '''
    def get_gate(self, key_j, inputs, W5):
        """
        Bi-linear formulation
        """
        a = tf.matmul(tf.expand_dims(key_j, 0), W5)
        b = tf.matmul(a, tf.transpose(inputs))
        gating = tf.reshape(b, [-1])
        print('gating: ', gating)
        return tf.sigmoid(gating)
    '''
    
    def get_gate(self, state_j, key_j, inputs, W1, W2, W3, W4, b1, b4):
        g1 = tf.matmul(state_j, W1)
        g2 = tf.matmul(tf.expand_dims(key_j, 0), W2)
        g3 = tf.matmul(inputs, W3)
        inter_gate = self._activation(g2 + g3 + b1)
        gating = tf.sigmoid(tf.matmul(inter_gate, W4) + b4)
        gating = tf.reshape(gating, [-1])
        print('gating', gating)
        return gating 
    

    def get_candidate(self, state_j, key_j, inputs, U, V, W):
        '''
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        '''
        key_V = tf.matmul(tf.expand_dims(key_j, 0), V)
        state_U = tf.matmul(state_j, U)
        inputs_W = tf.matmul(inputs, W)
        return self._activation(state_U + key_V + inputs_W)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
            # Split the hidden state into blocks (each U, V, W are shared across blocks).
            state = tf.split(state, num_or_size_splits=int(self._num_blocks), axis=1)

            # TODO: ortho init?
            normal_initializer = tf.random_normal_initializer(stddev=0.1)
            zero_initializer = tf.constant_initializer(0.0)
            U = tf.get_variable('U', [self._num_units_per_block, self._num_units_per_block])
            V = tf.get_variable('V', [self._num_units_per_block, self._num_units_per_block])
            W = tf.get_variable('W', [self._num_units_per_block, self._num_units_per_block])

            # TODO: layer norm?
            W1 = tf.get_variable('W1', [self._num_units_per_block, 20])
            W2 = tf.get_variable('W2', [self._num_units_per_block, 20])
            W3 = tf.get_variable('W3', [self._num_units_per_block, 20])
            W4 = tf.get_variable('W4', [20, 1])
            b1 = tf.get_variable('b1', [20])
            b4 = tf.get_variable('b4', [1])
            #W5 = tf.get_variable('W5', [self._num_units_per_block, self._num_units_per_block])

            next_states = []
            gates = []
            for j, state_j in enumerate(state): # Hidden State (j)
                key_j = self._keys[j]
                gate_j = self.get_gate(state_j, key_j, inputs, W1, W2, W3, W4, b1, b4)
                #print('gate_j: ', gate_j)
                #gate_j = tf.Print(gate_j, [gate_j], "value of gating: ", summarize=64)
                candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W)

                # Equation 4: h_j <- h_j + g_j * h_j^~
                # Perform an update of the hidden state (memory).
                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j

                # Equation 5: h_j <- h_j / \norm{h_j}
                # Forget previous memories by normalization.
                state_j_next = tf.nn.l2_normalize(state_j_next, -1, epsilon=1e-7) # TODO: Is epsilon necessary?
                gate_j = tf.reshape(gate_j, [-1,1])
                next_states.append(state_j_next)
                gates.append(gate_j)
            state_next = tf.concat(next_states, 1)
            gate_next = tf.concat(gates, 1)
            #print('gate_next: ', gate_next)
        return gate_next, state_next
