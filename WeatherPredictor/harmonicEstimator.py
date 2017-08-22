""" 

This file tries to estimate the next value in a harmonic signal using LSTM with a linear output layer

"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os

class lstmModel(object):
    """ 
    LSTM Model used to Predict a sinusiod signal
    """
    def __init__(self, isTraining = True):
        self.truncated_backprop_num = 90
        self.batch_size = 1000
        if isTraining is True:
            self.total_sample_num = self.batch_size*self.truncated_backprop_num*int(1e3)
        else:
            self.total_sample_num = int(1e3)

        # Network Parameters
        self.input_num = 1 # One sample of harmonic data
        self.state_size = 1 # Size of LSTM state
        if isTraining is True:
            self.isTraining = True # model is being used for training
        else:
            self.isTraining = False # model is being used for prediction only
            self.batch_size = 1
            self.truncated_backprop_num = self.total_sample_num

        self.buildModel()

    def buildModel(self):

        """ Build Graph"""
        # Graph Place Holders / Variables
        with tf.name_scope('inputs_targets'):
            input_batch_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_num],name='input_batch_placeholder')
            inputs_cur = tf.expand_dims(input_batch_placeholder,-1)
            target_batch_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_num],name='target_batch_placeholder')
            if self.isTraining:
                targets_cur = target_batch_placeholder[:, self.truncated_backprop_num//2:] # only use last half of truncated backprop length for errors
                targets_cur = tf.reshape(targets_cur,[-1, 1]) # Make column vector
            else:
                targets_cur = tf.reshape(target_batch_placeholder,[-1, 1]) # Make column vector

        self.feednames = {'input': input_batch_placeholder,
                          'target': target_batch_placeholder}


        # LSTM / Dynamic RNN
        with tf.name_scope('LSTM'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True, reuse= not self.isTraining)
            outputs, current_state = tf.nn.dynamic_rnn(cell,
                                                        inputs_cur,
                                                        initial_state=cell.zero_state(self.batch_size, dtype=tf.float32),
                                                        dtype=tf.float32)


        # Linear output layer
        W = tf.get_variable(name='W', shape=(self.state_size,1),dtype=tf.float32)
        b = tf.get_variable(name='b', dtype=tf.float32, shape=(1,1))
        tf.summary.histogram('W',W)
        tf.summary.scalar('b',tf.reduce_max(b))

        # Apply linear output layer
        with tf.name_scope('linear_output'):
            if self.isTraining: 
                outputslin = tf.reshape(outputs[:, round(self.truncated_backprop_num/2):], [-1, self.state_size],name='batchReshape') # reshape outputs for matrix multiply
            else:
                outputslin = tf.reshape(outputs, [-1, self.state_size],name='batchReshape') # reshape outputs for matrix multiply
            self.predictions = tf.add( tf.matmul(outputslin,W,name='linearW'), b, name='outputBias')  # running on all outputs in batch

        # Create Cost / Loss - Squared Error
        with tf.name_scope('cost'):
            errors = self.predictions - targets_cur
            costs =  (tf.square(errors))            
            self.costs=costs;



    def generateData(self, prediction_time=0.03):
        dt = 0.001
        frequency = 10 # Hz

        t = np.arange(self.total_sample_num) * dt
        inputs = np.array(np.sin(frequency * 2 * np.pi * t))
        targets =  np.array(np.sin(frequency * 2 * np.pi * (t + prediction_time)))

        # Batchify
        if self.isTraining is True:
            inputs = inputs.reshape((-1, self.truncated_backprop_num*self.batch_size))
            targets = targets.reshape((-1, self.truncated_backprop_num*self.batch_size))
        else:
            inputs = inputs.reshape((1, -1))
            targets = targets.reshape((1, -1))

        return (inputs, targets)


if __name__ == "__main__":
    LOGDIR = os.getcwd() + '/logs'
    # Turn on INFO based logging
    tf.logging.set_verbosity(tf.logging.WARN)
    print("Running")
    with tf.name_scope('Train'):
        with tf.variable_scope('MyModel', reuse=False):
            mtrain = lstmModel(isTraining=True)
            # Create Optimizer
            #train_step = tf.train.AdagradOptimizer(10).minimize(mtrain.costs)
            #train_step = tf.train.RMSPropOptimizer(learning_rate = 1.2).minimize(mtrain.costs)
            #train_step = tf.train.AdamOptimizer().minimize(mtrain.costs)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_step = tf.train.AdadeltaOptimizer(learning_rate = 1).minimize(mtrain.costs, global_step=global_step)
            tf.summary.scalar('cost_mean',tf.reduce_mean(mtrain.costs))
    with tf.name_scope('Pred'):
        with tf.variable_scope('MyModel', reuse=True):
            mpred = lstmModel(isTraining=False)
   
    summary_op = tf.summary.merge_all()

    pre_train_saver = tf.train.Saver();

    def load_pretrain(sess):
        pre_train_saver.restore(sess, LOGDIR)

    sv = tf.train.Supervisor(logdir=LOGDIR,
                             save_model_secs=1,
                             summary_op=None,
                             global_step=global_step,
                             )
    
    with sv.managed_session() as sess:


        cost_list = []
        pred_list = []

        inputs, targets = mtrain.generateData()
       
        for epoch in range(1):
            if sv.should_stop():
                break
            print('Running Epic', epoch)
            for batch_idx in range(inputs.shape[0]):
                if sv.should_stop():
                    break
                start_idx = batch_idx * mtrain.truncated_backprop_num
                stop_idx = start_idx + mtrain.truncated_backprop_num

                inputs_batch = inputs[batch_idx, :].reshape((mtrain.batch_size, -1))
                targets_batch = targets[batch_idx, :].reshape((mtrain.batch_size, -1))

                feed = {
                    mtrain.feednames['input']: inputs_batch,
                    mtrain.feednames['target']: targets_batch,
                        }

                _summary, _cost, _train_step, _predictions = sess.run(
                    [summary_op, mtrain.costs, train_step, mtrain.predictions],
                    feed_dict=feed)

                sv.summary_computed(sess, _summary)

                if batch_idx%100 == 0:
                    print("Step",batch_idx, "Cost", np.mean(_cost), "of", inputs.shape[0])


        # run trained model on all data
        print('Running Prediction Only')
        inputsp, targetsp = mpred.generateData()
        feed = {
                mpred.feednames['input']: inputsp,
                mpred.feednames['target']: targetsp,
                    }
        _predictionsAll = sess.run([mpred.predictions], feed_dict=feed)

    """
    Show Results
    """
    l1=plt.plot(inputsp.transpose(), label='input')
    l2=plt.plot(targetsp.transpose(), label='target')
    l3=plt.plot(_predictionsAll[0], label='pred')
    plt.axis([800,1000,-1, 1])
    plt.legend()
    plt.show()

    l1=plt.plot(inputsp.transpose(), label='input')
    l2=plt.plot(targetsp.transpose(), label='target')
    l3=plt.plot(_predictionsAll[0], label='pred')

    plt.axis([0,200,-1, 1])
    plt.legend()
    plt.show()

   




    

