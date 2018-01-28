import tensorflow as tf
import logging
import numpy as np
import pickle as pkl

class Classifier():
    def __init__(self, args):
        self.global_step = tf.Variable(0, trainable=False)
        self.model_setup(args)
        self._logger = logging.getLogger(__name__)

    def create_placeholders(self, args):
        self.input_x = tf.placeholder(
            dtype=tf.int64,
            shape=[None, args.max_seq_len],
            name='input_x')

        self.input_y = tf.placeholder(
            dtype=tf.int64,
            shape=[None, 9],
            name='input_y')

        self.dropout_keep_prob = tf.placeholder(
            dtype=tf.float32,
            name='dropout_keep_prob')

    def create_embedding_matrix(self, args):
        if args.embd_path is None:
            print("Random initialize character embeddings!!")
            np.random.seed(1234567890)
            embd_init = np.random.randn(args.vocab_size, args.embd_dim).astype(np.float32) * 1e-2
        else:
            embd_init = pkl.load(open(args.embd_path, 'rb'))
            assert embd_init.shape[0] == args.vocab_size \
                   and embd_init.shape[1] == args.embd_dim, \
                'Shapes between given pretrained embedding matrix and given settings do not match'
        with tf.variable_scope('embedding_matrix'):
            self.embedding_matrix = tf.get_variable(
                'embedding_matrix',
                [args.vocab_size, args.embd_dim],
                 initializer=tf.constant_initializer(embd_init))

    def create_encoder (self, inp, sequence_length, scope_name, args):
        with tf.variable_scope(scope_name):
            emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)

            if args.classifier_type == 'MT_CNN_1':
                emb_inp = tf.expand_dims(emb_inp, -1)  # [64, 212, 100, 1] means [batch_size, seq_max_len, emb_dim, 1]
                list_filter_size = [1, 2, 3, 4, 5]
                list_pool_output = []
                for fs in list_filter_size:
                    with tf.name_scope('conv_maxpool_{}'.format(fs)):
                        # filter_shape = [fs, int(emb_inp.shape[2]), 1, args.num_filters]
                        filter_shape = [fs, int(emb_inp.shape[2]), 1, 160]
                        # [1/2/3/4/5, 100, 1, 300] means [filter_size, emb_dim, 1, num_filters]
                        print('filter_shape: ' ,filter_shape)
                        conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        #conv_b = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b")
                        conv_b = tf.Variable(tf.constant(0.1, shape=[160]), name="b")
                        conv = tf.nn.conv2d(emb_inp, conv_W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                        # [64, 212, 1, 300] means [batch_size, seq_max_len, 1, num_filters]
                        conv = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                        conv = tf.squeeze(conv, 2)
                        print('squeeze conv: ', conv.shape)  # [64,212,300]
                        pooled = tf.reduce_max(tf.transpose(conv, [0, 2, 1]), [2])
                        # [64, 300] same for all filter_sizes
                        print('pool result: ', pooled)
                        list_pool_output.append(pooled)
                enc_state = tf.concat(list_pool_output, 1)   # [64, 1500]
                enc_state = tf.nn.dropout(enc_state, self.dropout_keep_prob)
                print('enc_state: ', enc_state)
                weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(self.input_y)[1]])   ### [64, 212, 15]
                self.gate_weights = weights
            elif args.classifier_type == 'MT_CNN_2':
                emb_inp = tf.expand_dims(emb_inp, -1)
                # [64, 212, 100, 1] means [batch_size, seq_max_len, emb_dim, 1]
                list_filter_size = [1, 3, 5]
                list_pool_output = []
                for fs in list_filter_size:
                    with tf.name_scope('conv_maxpool_{}'.format(fs)):
                        # filter_shape_1 = [fs, int(emb_inp.shape[2]), 1, args.num_filters]
                        filter_shape_1 = [fs, int(emb_inp.shape[2]), 1, 90]
                        # [1/2/3/4/5, 100, 1, 300] means [filter_size, emb_dim, 1, num_filters]
                        #filter_shape_2 = [fs, 1, args.num_filters, args.num_filters]
                        filter_shape_2 = [fs, 1, 90, 90]
                        # [1/2/3/4/5, 1, 300, 300]
                        conv_W_1 = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1), name="W1")
                        # conv_b_1 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b1")
                        conv_b_1 = tf.Variable(tf.constant(0.1, shape=[90]), name="b1")
                        conv_W_2 = tf.Variable(tf.truncated_normal(filter_shape_2, stddev=0.1), name="W2")
                        # conv_b_2 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b2")
                        conv_b_2 = tf.Variable(tf.constant(0.1, shape=[90]), name="b2")
                        conv1 = tf.nn.conv2d(emb_inp, conv_W_1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
                        # [64, 212, 1, 300] means [batch_size, seq_max_len, 1, num_filters]
                        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, conv_b_1), name="relu1")
                        pooled1 = tf.nn.max_pool(conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID',
                                                 name="pool1")  ### [64, 106/105/104/103/102, 1, 300]
                        conv2 = tf.nn.conv2d(pooled1, conv_W_2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
                        # [64, 106/103/100/97/94, 1, 300]
                        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, conv_b_2), name="relu2")
                        conv2 = tf.squeeze(conv2, 2)   # [64, 106/103/100/97/94, 300]
                        print('squeeze conv: ', conv2)
                        pooled2 = tf.reduce_max(tf.transpose(conv2, [0, 2, 1]), [2])
                        # [64, 300] same for all filter_sizes
                        print('pool result: ', pooled2)
                        list_pool_output.append(pooled2)
                enc_state = tf.concat(list_pool_output, 1)   # [64, 1500, 1] or [64, 1500]
                enc_state = tf.nn.dropout(enc_state, self.dropout_keep_prob)
                print('enc_state: ', enc_state)
                weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(self.input_y)[1]])   # [64, 212, 15]
                self.gate_weights = weights
            elif args.classifier_type == 'MT_GRU_S':
                # cell = tf.contrib.rnn.GRUCell(args.num_units)
                cell = tf.contrib.rnn.GRUCell(int(emb_inp.shape[2]))
                _, enc_state = tf.nn.dynamic_rnn(cell=cell,
                                                 inputs=emb_inp,
                                                 dtype=tf.float32,
                                                 sequence_length=sequence_length)   # enc_state [64, 1500]
                weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(self.input_y)[1]]) # [64, 212, 15]
                enc_state = tf.nn.dropout(enc_state, self.dropout_keep_prob)
                print('enc_state: ', enc_state)
                self.gate_weights = weights
            elif args.classifier_type == 'MT_GRU_Bi':
                # cell_fw = tf.contrib.rnn.GRUCell(args.num_units / 2)
                # cell_bw = tf.contrib.rnn.GRUCell(args.num_units / 2)
                cell_fw = tf.contrib.rnn.GRUCell(int(emb_inp.shape[2])/2)
                cell_bw = tf.contrib.rnn.GRUCell(int(emb_inp.shape[2])/2)
                _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                               cell_bw=cell_bw,
                                                               inputs=emb_inp,
                                                               sequence_length=sequence_length,
                                                               dtype=tf.float32)
                enc_state = tf.concat(enc_state, axis=1)   # enc_state [64, 1500]
                weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(self.input_y)[1]])
                enc_state = tf.nn.dropout(enc_state, self.dropout_keep_prob)
                print('enc_state: ', enc_state)
                self.gate_weights = weights
            elif args.classifier_type == 'BiGRU+maxpooling':
                # cell_fw = tf.contrib.rnn.GRUCell(args.num_units / 2)
                # cell_bw = tf.contrib.rnn.GRUCell(args.num_units / 2)
                cell_fw = tf.contrib.rnn.GRUCell(int(emb_inp.shape[2])/2)
                cell_bw = tf.contrib.rnn.GRUCell(int(emb_inp.shape[2])/2)
                enc_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                cell_bw=cell_bw,
                                                                inputs=emb_inp,
                                                                sequence_length=sequence_length,
                                                                dtype=tf.float32)
                enc_state = tf.concat(enc_states, axis=2)
                enc_state = tf.reduce_max(enc_state, axis=1)   # [64, 1500] or [64, 1, 1500]
                enc_state = tf.nn.dropout(enc_state, self.dropout_keep_prob)
                print('enc_state: ', enc_state)
                weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(self.input_y)[1]])
                self.gate_weights = weights
            elif args.classifier_type == 'self-attention':
                initializer = tf.contrib.layers.xavier_initializer()
                rnn_cell = tf.contrib.rnn.GRUCell(int(emb_inp.shape[2]))
                enc_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                   inputs=emb_inp,
                                                   dtype=tf.float32,
                                                   sequence_length=sequence_length)
                # Compute attention weight
                Ws1 = tf.get_variable("Ws1", shape=[int(emb_inp.shape[2]), 30],
                                      initializer=initializer)
                print("Ws1: ", Ws1)
                H_reshaped = tf.reshape(enc_outputs, [-1, int(emb_inp.shape[2])], name='H_reshaped')
                tanh_Ws1_time_H = tf.nn.tanh(tf.matmul(H_reshaped, Ws1), name="tanh_Ws1_time_H")
                print("tanh_Ws1_time_H: ", tanh_Ws1_time_H)
                Ws2 = tf.get_variable("Ws2", shape=[30, int(self.input_y.shape[1])], initializer=initializer)
                print("Ws2: ", Ws2)
                tanh_Ws1_time_H_and_time_Ws2 = tf.matmul(tanh_Ws1_time_H, Ws2, name="tanh_ws1_time_H_and_time_Ws2")
                print("tanh_Ws1_time_H_and_time_Ws2: ", tanh_Ws1_time_H_and_time_Ws2)
                self.A_T = tf.nn.softmax(tf.reshape(tanh_Ws1_time_H_and_time_Ws2, shape=[args.batch_size, args.max_seq_len, int(self.input_y.shape[1])], name="A_T_no_softmax"), dim=1, name="A_T")
                print("A_T: ", self.A_T)
                # Apply attention
                H_T = tf.transpose(enc_outputs, perm=[0, 2, 1], name="H_T")
                M_T = tf.matmul(H_T, self.A_T, name="M_T_no_transposed")
                # Compute penalization term
                A = tf.transpose(self.A_T, perm=[0, 2, 1], name="A")
                AA_T = tf.matmul(A, self.A_T, name="AA_T")
                identity = tf.reshape(
                    tf.tile(tf.diag(tf.ones(int(self.input_y.shape[1])), name="diag_identity"), [args.batch_size, 1],
                    name="tile_identity"), [args.batch_size, int(self.input_y.shape[1]), int(self.input_y.shape[1])],
                    name="identity")
                self.penalized_term = tf.square(tf.norm(AA_T - identity, ord='euclidean', axis=[1, 2], name="frobenius_norm"), name="penalized_term")
                print("penalized_term: ", self.penalized_term)
                weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(self.input_y)[1]])
                self.gate_weights = weights
                enc_state = M_T
                print('enc_state: ', enc_state)
            elif args.classifier_type == 'EXPERTCELL_GRU':
                from dynamic_memory_cell import DynamicMemoryCell
                from functools import partial
                from activations import prelu
                rnn_cell = tf.contrib.rnn.GRUCell(int(emb_inp.shape[2]))  # dim 100
                enc_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                   inputs=emb_inp,
                                                   dtype=tf.float32,
                                                   sequence_length=sequence_length)   # enc_outputs [64, 212, 100]
                num_blocks = int(self.input_y.shape[1])
                keys = [tf.get_variable('key_{}'.format(j), [int(emb_inp.shape[2])]) for j in range(num_blocks)]
                entity_cell = DynamicMemoryCell(num_blocks, int(emb_inp.shape[2]), keys, initializer=tf.random_normal_initializer(stddev=0.1),
                                                activation=partial(prelu, initializer=tf.constant_initializer(1.0)))
                gate_outputs, last_state = tf.nn.dynamic_rnn(entity_cell, enc_outputs,
                                                                  sequence_length=sequence_length,
                                                                  time_major=False, dtype=tf.float32)
                # gate_outputs [64, 212, 15]
                # last_state [64, 1500]
                weights = gate_outputs   # [64, 212, 15]
                last_state = tf.nn.dropout(last_state, self.dropout_keep_prob)
                enc_state = tf.split(last_state, int(self.input_y.shape[1]), 1)
                print('enc_state: ', enc_state)
                self.gate_weights = weights
            elif args.classifier_type == 'EXPERTCELL':
                from dynamic_memory_cell import DynamicMemoryCell
                from functools import partial
                from activations import prelu
                num_blocks = int(self.input_y.shape[1])
                keys = [tf.get_variable('key_{}'.format(j), [int(emb_inp.shape[2])]) for j in range(num_blocks)]
                entity_cell = DynamicMemoryCell(num_blocks, int(emb_inp.shape[2]), keys, initializer=tf.random_normal_initializer(stddev=0.1),
                                                activation=partial(prelu, initializer=tf.constant_initializer(1.0)))
                gate_outputs, last_state = tf.nn.dynamic_rnn(entity_cell, emb_inp,
                                                                  sequence_length=sequence_length,
                                                                  time_major=False, dtype=tf.float32)
                weights = gate_outputs
                last_state = tf.nn.dropout(last_state, self.dropout_keep_prob)
                #last_state_split = tf.stack(tf.split(last_state, int(self.input_y.shape[1]), 1), axis=0)
                #keyArray = tf.stack(keys, axis=0)
                #atten_coef = tf.nn.softmax(tf.matmul(keyArray, tf.transpose(keyArray))/tf.sqrt(float(int(emb_inp.shape[2]))))
                #atten_coef = tf.Print(atten_coef, [atten_coef], "attention coefficient: ", summarize=225)
                #atten_state = tf.reshape(tf.matmul(atten_coef, tf.reshape(last_state_split, [int(num_blocks), -1])), [int(num_blocks), -1, int(emb_inp.shape[2])])
                #enc_state = atten_state
                #print('enc_state: ', enc_state)
                enc_state = tf.split(last_state, int(self.input_y.shape[1]), 1)
                self.gate_weights = weights
            elif args.classifier_type == 'EXPERTCELL_CNN':
                from dynamic_memory_cell import DynamicMemoryCell
                from functools import partial
                from activations import prelu 
                list_filter_size = [1, 2, 3, 4, 5]
                list_conv_output = []
                for fs in list_filter_size:
                    filter_shape = [fs, int(emb_inp.shape[2]), 100]
                    conv_W = tf.Variable(tf.random_uniform(filter_shape, maxval=0.01), name="conv_W_1", dtype=tf.float32)
                    conv_b = tf.Variable(tf.zeros(shape=[100]), name="conv_b_1")
                    conv = tf.nn.conv1d(emb_inp, conv_W, stride=1, padding="SAME", name="conv1")
                    conv = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                    list_conv_output.append(conv)
                #states = tf.concat(list_conv_output, 2)
                states = tf.nn.dropout(tf.concat(list_conv_output, 2), self.dropout_keep_prob)
                print('CNN_states: ', states)
                num_blocks = int(self.input_y.shape[1])
                keys = [tf.get_variable('key_{}'.format(j), [500]) for j in range(num_blocks)]
                entity_cell = DynamicMemoryCell(num_blocks, int(states.shape[2]), keys, initializer=tf.random_normal_initializer(stddev=0.1),
                                                activation=partial(prelu, initializer=tf.constant_initializer(1.0)))
                gate_outputs, last_state = tf.nn.dynamic_rnn(entity_cell, states,
                                                                  sequence_length=sequence_length,
                                                                  time_major=False, dtype=tf.float32)
                print('last_state: ', last_state)
                weights = gate_outputs
                last_state = tf.nn.dropout(last_state, self.dropout_keep_prob)
                enc_state = tf.split(last_state, int(self.input_y.shape[1]), 1)
                print('enc_state: ', enc_state)
                self.gate_weights = weights
            else:
                raise 'Encoder type {} not supported'.format(args.classifier_type)
            #self._logger.info("Encoder done")
            return enc_state, weights

    def create_fclayers(self, enc_state, num_classes, scope, count): 
        enc_state = tf.contrib.layers.fully_connected(enc_state, 1000, scope=scope + 'fc0' + count)
        enc_state = tf.nn.tanh(enc_state)
        enc_state = tf.nn.dropout(enc_state, self.dropout_keep_prob)
        logits = tf.contrib.layers.fully_connected(
            inputs=enc_state,
            num_outputs=num_classes,
            activation_fn=None,
            scope=scope + 'fc1' + count)
        return logits  

    def mlp_weight_variable(self, shape):
        initial = tf.random_normal(shape=shape, stddev=0.01)
        mlp_W = tf.Variable(initial, name='mlp_W')
        return tf.clip_by_norm(mlp_W, 3.0) 

    def mlp_bias_variable(self, shape): 
        initial = tf.zeros(shape=shape)
        mlp_b = tf.Variable(initial, name='mlp_b')
        return mlp_b

    def get_loss(self, number, num_class, logits, label):
        label = tf.one_hot(label, num_class)
        print('logits: ', logits)
        print('label: ', label)
        #flat_logits = tf.reshape(logits, [-1, num_class])
        #flat_labels = tf.reshape(label, [-1, num_class])
        flat_logits = logits
        flat_labels = label

        eps = tf.constant(value=1e-10)
        flat_logits = flat_logits + eps

        softmax = tf.nn.softmax(flat_logits)

        if number == 0:
            coeffs = tf.constant([(0.5/0.8599), (0), (0.5/0.1401)])
        elif number == 1:
            coeffs = tf.constant([(0.5 / 0.8221), (0), (0.5 / 0.1779)])
        elif number == 2:
            coeffs = tf.constant([(0.33 / 0.8078), (0.33 / 0.0078), (0.33/0.1843)])
        elif number == 3:
            coeffs = tf.constant([(0.33 / 0.8159), (0.33 / 0.0062), (0.33/0.1779)])
        elif number == 4:
            coeffs = tf.constant([(0.33 / 0.5477), (0.33 / 0.0618), (0.33/0.3905)])
        elif number == 5:
            coeffs = tf.constant([(0.33 / 0.7301), (0.33 / 0.0772), (0.33/0.1927)])
        elif number == 6:
            coeffs = tf.constant([(0.33 / 0.8414), (0.33 / 0.0006), (0.33/0.158)])

        cross_entropy = -tf.reduce_sum(tf.multiply(flat_labels * tf.log(softmax + eps), coeffs), reduction_indices=[1])
        #loss_beta=0.01
        #loss_L2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.vars if 'w' in v.name ]) * loss_beta
        cost = tf.reduce_mean(cross_entropy, name='cost')
        return cost

    def model_setup(self, args): 
        self.create_placeholders(args)
        self.create_embedding_matrix(args)
        np.random.seed(12345)
        mask = tf.sign(self.input_x) 
        seqlen = tf.to_int64(tf.reduce_sum(mask, axis=1))
        enc_state, self.gate_weights = self.create_encoder(
            inp=self.input_x,
            sequence_length=seqlen,
            scope_name='encoder',
            args=args)
        
        
        # gate_weights [64, 212, 15]
        self.loss_reg_l1 = tf.reduce_mean(tf.reduce_sum(self.gate_weights, axis=1))
        self.loss_reg_diff = tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.gate_weights[:, 1:, :] - self.gate_weights[:, :-1, :]),
                          axis=1))

        self.losses = []
        self.accuracies = []
        self.predictions = []
        for count in range(0, int(self.input_y.shape[1])):
            if args.classifier_type in ['EXPERTCELL_GRU', 'EXPERTCELL', 'EXPERTCELL_CNN']:
                state = enc_state[count]
                #state = enc_state[count,:,:]
            else:
                state = enc_state
            if count==8:
                class_number = 5
            else:
                class_number = 3
            logits = self.create_fclayers(state, class_number, 'fclayers', str(count))
            
            """
            mlp_W_1 = self.mlp_weight_variable([int(state.shape[1]), 30])
            mlp_b_1 = self.mlp_bias_variable([30])
            mlp_inter = tf.matmul(state, mlp_W_1) + mlp_b_1
            mlp_W_2 = self.mlp_weight_variable([30, class_number])
            mlp_b_2 = self.mlp_bias_variable([class_number])
            logits = tf.matmul(mlp_inter, mlp_W_2) + mlp_b_2
            print('logits: ', logits)
            """

            self.prob = tf.nn.softmax(logits)
            self.pred = tf.argmax(logits, axis=1)
            #self.pred = tf.Print(self.pred, [self.pred], "prediction: ", summarize=64)
            self.predictions.append(self.pred)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.input_y[:, count]), tf.float32))
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y[:, count], logits=self.prob))
            #self.loss = self.get_loss(count, class_number, logits, self.input_y[:, count])
            
            self.losses.append(self.loss)
            self.accuracies.append(self.accuracy)

        with tf.name_scope('loss'):
            original_loss = tf.reduce_mean(self.losses)
            penalize_loss = self.loss_reg_l1 / args.max_seq_len * args.sparsity + self.loss_reg_diff / args.max_seq_len * args.sparsity * args.coherent
            #original_loss = tf.Print(original_loss, [original_loss], "original_loss: ")
            #penalize_loss = tf.Print(penalize_loss, [penalize_loss], "penalize_loss: ")
            self.loss_average = original_loss + penalize_loss
        with tf.name_scope('accuracy'):
            self.acc_average = tf.reduce_mean(self.accuracies)
        
        '''
        #Self-attention loss
        w4 = tf.reshape(tf.tile(tf.Variable(tf.random_normal([100, 30], stddev=0.01)), [args.batch_size, 1]), [args.batch_size, 100, 30])
        b4 = tf.Variable(tf.zeros([args.batch_size, int(self.input_y.shape[1]), 30]))
        layer = tf.add(tf.matmul(tf.transpose(enc_state, perm=[0, 2, 1], name="M"), w4), b4)
        print('layer: ', layer)
        w3 = tf.reshape(tf.tile(tf.Variable(tf.random_normal([30, 2], stddev=0.01)), [args.batch_size, 1]), [args.batch_size, 30, 2])
        b3 = tf.Variable(tf.zeros([args.batch_size, int(self.input_y.shape[1]), 2]))
        logits = tf.add(tf.matmul(layer, w3), b3)
        #self.pred = tf.argmax(logits, axis=2)
        self.losses = []
        self.accuracies = []
        self.predictions = []
        for count in range(0, int(self.input_y.shape[1])):
            self.pred = tf.argmax(logits[:,count,:], axis=1)
            self.predictions.append(self.pred)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.input_y[:, count]), tf.float32))
            self.accuracies.append(self.accuracy)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y[:, count], logits=logits[:,count,:]))
            self.losses.append(self.loss)
        self.loss_average = tf.reduce_mean(self.losses)
        self.acc_average = tf.reduce_mean(self.accuracies)
        #self.loss_average = self.loss_average + 0.1*tf.sigmoid(tf.reduce_mean(penalized_term))
        '''

        with tf.name_scope('backpropagation'):
            learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step,
                                                       args.decay_steps,
                                                       args.decay_rate,
                                                       staircase=True)
            optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss_average, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, args.grad_clip)
            self.train_op = optim.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=122)
        
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
             shape = variable.get_shape()
             print(shape)
             variable_parametes = 1
             for dim in shape:
                 variable_parametes *= dim.value
             total_parameters += variable_parametes
        print('total_parameter: ', total_parameters)

        tf.summary.scalar('average_loss', self.loss_average)
        tf.summary.scalar('loss_reg_l1', self.loss_reg_l1)
        tf.summary.scalar('loss_reg_diff', self.loss_reg_diff)
        tf.summary.scalar('average_accuracy', self.acc_average)
        self.merged = tf.summary.merge_all()

    def step(self, mode='train'): 
        if mode == 'train':
            ops = (self.train_op, self.global_step, self.merged, self.loss_average,
                    self.acc_average, self.losses, self.accuracies, self.gate_weights)
        elif mode == 'valid': 
            ops = (self.global_step, self.merged, self.loss_average, self.acc_average, self.losses, self.accuracies, self.predictions)
        return ops
