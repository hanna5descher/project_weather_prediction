
"""
This function builds and trains a model in Tensorflow to predict the temperature 24 hours from the current temperature.
The input is hourly weather data from NOAA, consisting of temperature, realtive humidity, dewpoint, pressure.
The output is a set of probabilities for the temperature 24 hours from current input, as degrees from -10 to 110 degC
"""

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

"""
Class: Build model utilizing LSTM and multilayer dense networks
Input consists of [temp humidity dewpoint pressure] * number of stations
"""
class tempModel(object):
    def __init__(self, num_of_classes, input_length,
                       truncated_backprop_num = 2*24,  isTraining = True, 
                       batch_size = 1000, stations = [93785, 63821, 13713], state_size = 10,
                       dropout_keep_prob = .9, num_rnn_layers = 10):
        self.isTraining = isTraining
        self.truncated_backprop_num = truncated_backprop_num
        self.batch_size = batch_size
        self.inputLength = input_length
        self.state_size = state_size
        self.dropout_keep_prob = dropout_keep_prob
        self.num_rnn_layers = num_rnn_layers
        self.num_of_classes = num_of_classes
        self.stations = stations

        if not self.isTraining:
            self.batch_size = 1
        # Build model
        self._buildModel()

    def _buildModel(self):
        # Graph Place Holders / Variables: to be filled in by feed (see below)
        with tf.name_scope('inputs'):
            input_batch_PH = tf.placeholder(tf.float32, [self.batch_size, 
                                                         self.truncated_backprop_num,
                                                         self.inputLength],
                                            name='input_batch')
        with tf.name_scope('targets'):
            target_batch_PH = tf.placeholder(tf.float32, [self.batch_size,
                                                          self.truncated_backprop_num,
                                                          1],
                                             name='target_batch')
            if self.isTraining:
                target_batch_cur = target_batch_PH[:,-1:,:] # Take only the last hour for cost calc
                target_batch_cur = tf.one_hot(tf.to_int32(target_batch_cur), depth=self.num_of_classes)
            else:
                target_batch_cur = tf.one_hot(tf.to_int32(target_batch_PH), depth=self.num_of_classes) # Take everything when running predictions

        self.inputs = input_batch_PH
        self.targets = target_batch_PH
        # LSTM / Dynamic RNN
        with tf.name_scope('LSTM'):
            doReuse = not self.isTraining # if making prediction, reuse the learned network variables
            cells=[]
            if self.isTraining:
                for _ in range(self.num_rnn_layers):
                    cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True, reuse=doReuse)
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) # regularization: dropping output in between LSTM cells
                    cells.append(cell)
            else:
                for _ in range(self.num_rnn_layers):
                    cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True, reuse=doReuse)
                    cells.append(cell)
                
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True) # linking between the cells
            outputs_batch, _ = tf.nn.dynamic_rnn(cell, input_batch_PH, dtype=tf.float32) # automatically unrolls network for backpropagation 
            outputs = tf.reshape(outputs_batch,[-1, self.state_size])

        with tf.name_scope('LinearLayer'):
            W = tf.get_variable('W', shape=(self.state_size, self.num_of_classes),dtype=tf.float32)
            b = tf.get_variable('b',shape=(1,self.num_of_classes))
            tf.summary.histogram('W',W)
            temp_pred_logits = tf.matmul(outputs, W) + b
            self.temp_pred_prob = tf.nn.softmax(temp_pred_logits)

        if self.isTraining:

            with tf.name_scope('cost_calc'):
                costs = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(temp_pred_logits,(self.batch_size,self.truncated_backprop_num,self.num_of_classes))[:,-1:,:],
                                                                labels=target_batch_cur)
                self.cost_mean = tf.reduce_mean(costs)

"""
Main Script
"""
if __name__ == "__main__":
    os.system('cls')
    LOGDIR = os.getcwd() + '/logs'
    tf.logging.set_verbosity(tf.logging.INFO) # enable log output
    print("Preparing data...")

    ### Set Parameters ###
    num_of_train_steps = int(1)
    min_temp = -10
    max_temp = 110
    num_of_classes = len(range(min_temp,max_temp)) # model temperatures for -10 F to 110 F (observed range: -2 to 106 F)
    num_rnn_layers = 2 # number of LSTMs (could be optimized)
    learning_rate = 0.5 
    batch_size = 1000 # size mini-batch (batches run in parallel)
    state_size = 100
    truncated_backprop_num = 24*1
    num_of_prediction = 24 # number of data points to predict

    # Load station data (see processStationData.py)    
    dirStation = 'dataStation'    
    dataStationCent = pd.read_csv(os.path.join(dirStation,'dataStationCentroids.csv'))
    wbanTargetStation = dataStationCent['WBAN'][0] # home station: Chapel Hill, NC
    wbanStation = dataStationCent['WBAN'].values

    # Load weather data (see createTrainingData.py)
    dirWeather = 'dataWeather'
    train_data = pd.read_hdf(os.path.join(dirWeather,'dataTrainingWeatherSOI.hdf')) # load dataframe

    # Select weather data from 10 center stations + home station (total 11 stations)
    b = None
    for station in wbanStation:
        if b is None:
            b = train_data.columns.to_series().str.endswith('_' + str(station))
        else:
            b = b | train_data.columns.to_series().str.endswith('_' + str(station))
    train_data = train_data.loc[:,b]

    # Fill in missing data
    b = train_data.columns.to_series().str.startswith('HourlyPrecip') # NaN seems to mean no rain, so replace with 0
    train_data.loc[:,b] = train_data.loc[:,b].fillna(0)
    train_data.fillna(train_data.mean(),inplace=True) # replace all other missing data with mean of feature
    train_data.fillna(0,inplace=True) # remove instances of weather stations with all NaNs
    stations = np.unique(np.array(train_data.columns.to_series().str.split('_').tolist())[:,1]) # get station WBAN out of column name (last part is digits)
    measurement_types = np.unique(np.array(train_data.columns.to_series().str.split('_').tolist())[:,0]) # get measurement out of column name (first part is measurement name from NOAA)

    # Get target Data
    target_data = train_data.loc[:,'DryBulbFarenheit_' + str(wbanTargetStation)].values
    target_data = target_data - min_temp
    target_data = np.round(target_data)
    target_data = np.expand_dims(target_data, axis=2)

    # Normalize input data
    input_data = np.empty(train_data.shape)
    for type in measurement_types:
        print('Normalizing measurement: ', type)
        b = train_data.columns.to_series().str.startswith(type)
        maxVal = train_data.loc[:,b].max().max() # global max
        minVal = train_data.loc[:,b].min().min() # global min
        input_data[:,b] = (train_data.loc[:,b].values - minVal) / (maxVal - minVal) 
    
    
    """
    TensorFlow using LSTM (Long Short-Term Memory) Network
    """
    print("Running TensorFlow")

    # Create mini-batches
    def getFeeder(model):
        truncated_backprop_num = model.truncated_backprop_num
        batch_size = model.batch_size
        total_size = batch_size*truncated_backprop_num
        
        if batch_size > 1: # if training
            randIndexs = np.random.randint(0,input_data.shape[0]-truncated_backprop_num-num_of_prediction*2,total_size) # randomize batch start index
            inputs = np.zeros((batch_size,truncated_backprop_num,input_data.shape[1]),dtype=np.float32)
            targets = np.zeros((batch_size,truncated_backprop_num,1),dtype=np.float32)
            for batch in np.arange(batch_size):
                selection = np.arange(randIndexs[batch],randIndexs[batch]+truncated_backprop_num)
                inputs[batch,:,:] = input_data[selection] # [batch number, weather data[time,measurement]]
                targets[batch,:,:] = np.expand_dims(target_data[selection+num_of_prediction],axis=0)
        else:  # for prediction
            testIndex = np.arange(-truncated_backprop_num*2,-truncated_backprop_num*2+truncated_backprop_num)
            inputs = np.zeros((1,truncated_backprop_num,input_data.shape[1]),dtype=np.float32)
            targets = np.zeros((1,truncated_backprop_num,1),dtype=np.float32)
            inputs[0,:,:] = input_data[testIndex]
            targets[0,:,:] = np.expand_dims(target_data[testIndex+num_of_prediction],axis=0)
        
        # create a dictionary for placeholder values 
        feed = {
                    model.inputs: inputs,
                    model.targets: targets
                }

        return feed, inputs, targets

    # Build training model
    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=False):
            modelTrain = tempModel(num_of_classes, input_data.shape[1], isTraining=True, state_size=state_size, batch_size=batch_size, truncated_backprop_num=truncated_backprop_num,
                                   num_rnn_layers=num_rnn_layers, stations=stations)
            global_step = tf.Variable(0, name='global_step', trainable=False) # counts the number of optimizer iteration
            train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=modelTrain.cost_mean, global_step=global_step)
            tf.summary.scalar('cost_mean',modelTrain.cost_mean)
            tf.summary.histogram('prediction_prob',modelTrain.temp_pred_prob[-1,:])

    # Build Prediction Model
    with tf.name_scope('Pred'):
        with tf.variable_scope('Model', reuse=True):
            modelPred = tempModel(num_of_classes, input_data.shape[1], isTraining=False, state_size=state_size, batch_size=1, truncated_backprop_num=truncated_backprop_num,
                                  num_rnn_layers=num_rnn_layers, stations=stations)

    # Merge all summaries
    summary = tf.summary.merge_all()

    sv = tf.train.Supervisor(logdir=LOGDIR,
                             save_model_secs=60,
                             summary_op=None,
                             global_step=global_step)
     
    # Run TensorFlow
    with sv.managed_session() as sess:
        # Train model
        for step in range(num_of_train_steps):
            if sv.should_stop():
                break

            feed, inputs, targets = getFeeder(modelTrain)   

            _summary, _train_step, _predictions, _cost_mean = sess.run(
                [summary, train_step, modelTrain.temp_pred_prob, modelTrain.cost_mean],
                feed_dict=feed)            

            if step % 10 == 0:
                sv.summary_computed(sess, _summary)
                print("Step", step, 'Cost', _cost_mean, 'of', num_of_train_steps)

        # Use trained model for predictions
        #feed = getFeeder(modelPred)      
        #_predictions = sess.run([modelPred.temp_pred_prob], feed_dict=feed)

        p =_predictions.reshape((batch_size,24,-1))
        maxPrediction = np.empty((batch_size))
        for batch in np.arange(batch_size):
            maxPrediction[batch] = np.argmax(p[batch,-1,:]) + min_temp
        t = targets[:,-1,0] + min_temp
        indSort = np.argsort(t)
        targetSorted = np.sort(t)
        maxPrediction = maxPrediction[indSort]        
        plt.plot(maxPrediction,label='model prediction')
        plt.plot(targetSorted,label = 'actual data')
        plt.legend()
        plt.xlabel('number of mini batchs')
        plt.ylabel('temperature (F)')
        plt.show()
            
