"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('sequence', help='sequence datasets')
    parser.add_argument('--name',choices=['ptb'])
    parser.add_argument('--data_dir',type=str,default='data/')
    parser.add_argument('--numdp',type=interval(int),default=1e10)
    parser.add_argument('--numdp_balanced',action='store_true')
    parser.add_argument('--numdp_test',type=interval(int),default=1e10)
    parser.add_argument('--seed',type=interval(int),default=0)

    parser.add_argument('--test_on_train',action='store_true')

def init(args):

    global train
    global train_numdp
    global train_X
    global train_Y
    global test
    global test_numdp
    global test_X
    global test_Y
    global dimX
    global dimY

    import tensorflow as tf
    import tflearn
    import numpy as np
    import random

    random.seed(args['seed'])
    np.random.seed(args['seed'])

    if args['name']=='ptb':
        import urllib
        urlroot='https://raw.githubusercontent.com/wojzaremba/lstm/master/data/'
        for file in ['ptb.test.txt','ptb.train.txt','ptb.valid.txt']:
            urllib.urlretrieve(urlroot+file,args['data_dir']+'/'+file)
        train, _ , test, _ = ptb_raw_data(data_path=args['data_dir'])
        train_X,train_Y=ptb_producer(train, 1, 1000, name=None)
        test_X,test_Y=ptb_producer(test, 1, 1000, name=None)
        dimX=train_X.get_shape()
        dimY=1000
        print('train_X',train_X.shape)
        print('train_Y',train_Y.shape)
    else:
        raise ValueError(args['name'] + ' not yet implemented')

    print('train_X=',train_X.shape)
    print('train_Y=',train_Y.shape)

    train_numdp = min(args['numdp'],train_X.shape[0])
    test_numdp = min(args['numdp_test'],test_X.shape[0])

    print('dimX=',dimX)
    print('dimY=',dimY)

    # training data

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    np.random.shuffle(train_X)

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    np.random.shuffle(train_Y)

    if args['test_on_train']:
        test_X=train_X
        test_Y=train_Y

    if args['numdp_balanced']:
        Yargmax = np.argmax(train_Y,axis=1)
        numdp_per_class=train_numdp/dimY
        allIndices=[]
        for i in range(0,10):
            arr,=np.where(Yargmax==i)
            allIndices += np.ndarray.tolist(arr[:numdp_per_class])
        train_X = train_X[allIndices]
        train_Y = train_Y[allIndices]

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        np.random.shuffle(train_X)

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        np.random.shuffle(train_Y)

    else:
        train_X = train_X[0:train_numdp,...]
        train_Y = train_Y[0:train_numdp]

    Id = np.array(range(0,train_numdp))
    train=tf.data.Dataset.from_tensor_slices((np.float32(train_X),np.float32(train_Y),Id))

    # testing data

    test_X = test_X[0:test_numdp,...]
    test_Y = test_Y[0:test_numdp]
    Id = np.array(range(0,test_numdp))
    test=tf.data.Dataset.from_tensor_slices((np.float32(test_X),np.float32(test_Y),Id))



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import collections
import os
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  print('file=',collections.__file__)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

