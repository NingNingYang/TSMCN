# -*- coding:utf-8 -*-
import tensorflow as tf
import time
import datetime
import json
import os
import data_helpers
import argparse
import numpy as np
from data_helpers import DataSelector
from model import Classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#np.set_printoptions(threshold='nan')

parser = argparse.ArgumentParser()
parser.add_argument('--classifier_type', default='EXPERTCELL_CNN', type=str, help='type of classifier')
parser.add_argument('--eval_every', default=50, type=int, help='evaluate the model every 50 steps')
parser.add_argument('--max_seq_len', default=212, type=int, help='max sequence length')
parser.add_argument('--embd_path', default=None, type=str, help='input path of pre-trained word embedding')
parser.add_argument('--vocab_size', default=7783, type=int, help='the vocabulary size')
parser.add_argument('--embd_dim', default=200, type=int, help='embedding size of character')
parser.add_argument('--tag_dim', default=100, type=int, help='embedding size of tag')
parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
parser.add_argument('--num_filters', default=300, type=int, help='number of filters in CNN')
parser.add_argument('--num_units', default=200, type=int, help='number of units in RNN')
parser.add_argument('--drop_rate', default=0.5, type=float, help='dropout rate')
parser.add_argument('--sparsity', default=0, type=float, help='loss lambda1')
parser.add_argument('--coherent', default=2.0, type=float, help='loss lambda2')
parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate')
parser.add_argument('--decay_steps', default=50, type=int, help='decay every 50 steps with decay rate:lr_decay_rate')
parser.add_argument('--decay_rate', default=0.95, type=float, help='decay learning rate')
parser.add_argument('--grad_clip', default=3, type=int, help='clip big gradient')
parser.add_argument('--num_epoches', default=50, type=int, help='num of epoch')
parser.add_argument('--checkpoint_every', default=50, type=int, help='Save model after this many steps (default: 100)')

args = parser.parse_args()
args_hash = vars(args)
print args_hash


class CONFIG(object):
    classifier_type = args_hash['classifier_type']
    eval_every = args_hash['eval_every']
    max_seq_len = args_hash['max_seq_len']
    embd_path = args_hash['embd_path']
    vocab_size = args_hash['vocab_size']
    embd_dim = args_hash['embd_dim']
    tag_dim = args_hash['tag_dim']
    batch_size = args_hash['batch_size']
    num_filters = args_hash['num_filters']
    num_units = args_hash['num_units']
    drop_rate = args_hash['drop_rate']
    sparsity = args_hash['sparsity']
    coherent = args_hash['coherent']
    learning_rate = args_hash['learning_rate']
    decay_rate = args_hash['decay_rate']
    decay_steps = args_hash['decay_steps']
    grad_clip = args_hash['grad_clip']
    num_epoches = args_hash['num_epoches']
    checkpoint_every = args_hash['checkpoint_every']


def main():
    args = CONFIG()
    kwargs = {
        "rnd": 1234,
        "train_data_dir": './data/feature_train.csv',
        "test_data_dir": './data/feature20170609_original.csv',
        "slt_cols": ['InjIden', 'Employ', 'ConfrmLevel', 'AppPay', 'WorkTime', 'WorkPlace', 'AssoPay', 'HaveMedicalFee', 'Identity']
    }
    dataset = DataSelector(kwargs['train_data_dir'], kwargs['rnd'], kwargs['slt_cols'])
    x = dataset.getInputs()
    y = dataset.getLabels()
    vocab = dataset.getVocab()
    # Write vocabulary
    jsObj = json.dumps(vocab)
    fileObject = open("./data/vocab", 'w')
    fileObject.write(jsObj)
    fileObject.close()

    y = y + 1
    #y = np.where(y < 2, 0, 1)
    #print x
    print y
    print('x shape: ', x.shape)
    print('y shape: ', y.shape)
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    x_train, x_dev = x[int(y.shape[0] / 6):], x[:int(y.shape[0] / 6)]
    y_train, y_dev = y[int(y.shape[0] / 6):], y[:int(y.shape[0] / 6)]
    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            model = Classifier(args)

            # writing directory
            #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs3", args.classifier_type, str(args.sparsity)))
            #print("Writing to {}\n".format(out_dir))

            # summary & checkpoint
            #train_summary_dir = os.path.join(out_dir, "summaries", "train")
            #dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            #train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            #dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            #checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            #checkpoint_prefix = os.path.join(checkpoint_dir, "model.chkp")
            #if not os.path.exists(checkpoint_dir):
            #    os.makedirs(checkpoint_dir)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            '''
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print(shape)
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            print('total_parameter: ', total_parameters)
            '''
            #gating_path = './gating3/gating_3_' + args.classifier_type + '.txt'
            #f1 = open(gating_path, 'w')

            def train_step(x_batch, y_batch):
                sentence = ''
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: args.drop_rate
                }
                ops = model.step(mode='train')
                _, step, summaries, ave_loss, ave_accuracy, _, each_accuracy, gate_outputs = sess.run(
                    ops, feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, ave_loss, ave_accuracy))
                #if args.classifier_type == 'EXPERTCELL_GRU':
                #    if step % 200 == 0:
                #        array_x_batch = np.array(x_batch)
                #        x_mask = np.sign(array_x_batch)
                #        x_sequence_length = np.sum(x_mask, axis=1)
                #        new_vocab = {v: k for k, v in vocab.items()}
                #        for i in range(0, array_x_batch.shape[0]):
                #            index_sentence = array_x_batch[i]
                #            for j in range(0, x_sequence_length[i]):
                #                sentence = sentence + new_vocab[index_sentence[j]]
                #            sentence = sentence + "\n"
                #        f1.write(sentence)
                #        f1.write(str(x_sequence_length))
                #        f1.write(str(gate_outputs))
                #train_summary_writer.add_summary(summaries, step)
                return step 

            def dev_step(x_batch,  y_batch):
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: 1.0
                }
                ops = model.step(mode='valid')
                step, summaries, ave_loss, ave_accuracy, _, each_accuracy, pred = sess.run(ops, feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #dev_summary_writer.add_summary(summaries, step)
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, ave_loss, ave_accuracy))
                print("Accuracy of each category:")
                np_y_batch = np.asarray(y_batch)
                for class_num in range(0, len(each_accuracy)):
                    print("{}: step {}, acc {}".format(class_num, step, each_accuracy[class_num]))
                    print('accuracy score: ')
                    print(accuracy_score(y_batch[:, class_num], pred[class_num]))
                    print(classification_report(y_batch[:, class_num], pred[class_num]))
                    print('******************************************************************')
                #print('prediction: ', pred)

            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), args.batch_size, args.num_epoches)
            #dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), args.batch_size, 1)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                if len(x_batch) == 64:
                    current_step = train_step(x_batch, y_batch)
                if current_step % args.eval_every == 0:
                    print("\nEvaluation:")
                    #dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), args.batch_size, 1)
                    #for dev_batch in dev_batches:
                    #    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    #    if len(x_dev_batch) == 64:
                    dev_step(x_dev, y_dev)
                #if current_step % args.checkpoint_every == 0:
                #    path = model.saver.save(sess, checkpoint_prefix, global_step=current_step)
                #    print("Saved model checkpoint to {}\n".format(path))
            #f1.close()
    print("Optimization Finished!")

if __name__ == "__main__":
    main()
