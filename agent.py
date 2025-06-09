import tensorflow as tf 
from tensorflow.compat.v1.nn.rnn_cell import LSTMStateTuple

#Define the agent structure network
class Network():
    """
    Description: 
        Defines a deep recurrent neural network agent for navigation tasks. The model processes sequences of 
        velocity and angular velocity inputs using an LSTM whose initial hidden and cell states are derived 
        from place and head cell activations. A linear layer then decodes these hidden states into 
        predicted place cell and head cell activity distributions.

    --------- Parameters ---------
    @param session: A TensorFlow session used to execute operations and manage model state.
    @param lr: Learning rate used for RMSProp optimization.
    @param hu: Number of hidden units in the LSTM.
    @param lu: Number of linear layer units used for decoding.
    @param place_units: Number of place cell units for encoding spatial information.
    @param head_units: Number of head direction cell units.
    @param clipping: Value used for gradient clipping to stabilize training.
    @param weightDecay: L2 weight decay regularization coefficient.
    @param batch_size: Number of trajectories processed per training batch.
    @param num_features: Number of input features (e.g., velocity, angular velocity components).
    @param n_steps: Number of timesteps in each trajectory.

    -------- Returns --------
    None
    """
    def __init__(self, session, lr, hu, lu, place_units, head_units, clipping, weightDecay, batch_size, num_features, n_steps):
        self.sess = session
        self.epoch = tf.Variable(0, trainable=False)
        #HYPERPARAMETERS
        self.learning_rate = lr #learning rate
        self.Hidden_units = hu #number of hidden units
        self.LinearLayer_units = lu #number of linear layer units
        self.PlaceCells_units = place_units #number of place cell units
        self.HeadCells_units = head_units #number of head cell units
        self.clipping = clipping #gradient clipping
        self.weight_decay = tf.constant(weightDecay, dtype=tf.float32) #weight decay
        self.batch_size = batch_size #batch size
        self.num_features = num_features #number of features

        self.buildNetwork()
        self.buildTraining()
        self.buildTensorBoardStats()

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver = tf.compat.v1.train.Saver()
        self.file = tf.compat.v1.summary.FileWriter("tensorboard/", self.sess.graph)

    def buildNetwork(self):
        """
        Description: 
            Builds the architecture of the agent, which includes:
            - Input placeholders for trajectory data and dropout.
            - LSTM cell initialized using learned transformations of place and head cell distributions.
            - A linear decoder layer applied to LSTM outputs across all time steps.
            - Separate output layers for predicting place and head cell activity from the linear decoder layer.
        -------- Parameters --------
        None
        -------- Returns --------
        None
        """
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, 100, self.num_features], name="input") #placeholder for input

        self.keepProb = tf.compat.v1.placeholder(tf.float32, name="keep_prob") #placeholder for dropout probability

        self.placeCellGround = tf.compat.v1.placeholder(tf.float32, shape=[None, self.PlaceCells_units], name="Ground_Truth_Place_Cell") #placeholder for ground-truth of place cell
        self.headCellGround = tf.compat.v1.placeholder(tf.float32, shape=[None, self.HeadCells_units], name="Ground_Truth_Head_Cell") #placeholder for ground-truth of head cell

        with tf.compat.v1.variable_scope("LSTM_initialization"):
            #Initialize the Hidden state and Cell state of the LSTM unit using feeding the Ground Truth Distribution at timestep 0. Both have size [batch_size, Hidden_units]
            self.Wcp = tf.compat.v1.get_variable("Initial_state_cp", [self.PlaceCells_units,self.Hidden_units], initializer=tf.keras.initializers.GlorotUniform())
            self.Wcd = tf.compat.v1.get_variable("Initial_state_cd", [self.HeadCells_units,self.Hidden_units],  initializer=tf.keras.initializers.GlorotUniform())
            self.Whp = tf.compat.v1.get_variable("Hidden_state_hp",  [self.PlaceCells_units,self.Hidden_units], initializer=tf.keras.initializers.GlorotUniform())
            self.Whd = tf.compat.v1.get_variable("Hidden_state_hd",  [self.HeadCells_units,self.Hidden_units],  initializer=tf.keras.initializers.GlorotUniform())

            #Compute self.hidden_state 
            self.hidden_state = tf.matmul(self.placeCellGround, self.Whp) + tf.matmul( self.headCellGround, self.Whd)
            #Compute self.cell_state
            self.cell_state = tf.matmul(self.placeCellGround, self.Wcp) + tf.matmul( self.headCellGround, self.Wcd)

            #Store self.cell_state and self.hidden_state tensors as elements of a single list.
            #If is going to be timestep=0, initialize the hidden and cell state using the Ground Truth Distributions. 
            #Otherwise, use the hidden state and cell state from the previous timestep passed using the placeholders  

            self.LSTM_state = LSTMStateTuple(self.hidden_state, self.cell_state)  
        with tf.compat.v1.variable_scope("LSTM"):
            self.lstm_cell = tf.keras.layers.LSTMCell(self.Hidden_units, name="LSTM_Cell")
            rnn_layer = tf.keras.layers.RNN(self.lstm_cell, return_sequences=True, return_state=True)
            self.output, h, c = rnn_layer(self.X, initial_state=self.LSTM_state)
            self.hidden_cell_statesTuple = (h, c)
        with tf.compat.v1.variable_scope("Linear_Decoder"):
            self.W1 = tf.compat.v1.get_variable("Weights_LSTM_LinearDecoder", [self.Hidden_units, self.LinearLayer_units], initializer=tf.keras.initializers.GlorotUniform())
            self.B1 = tf.compat.v1.get_variable("Biases_LSTM_LinearDecoder", [self.LinearLayer_units], initializer=tf.keras.initializers.GlorotUniform())
            
            #we can't feed a tensor of shape [10,100,128] to the linear layer. We treat each timestep in every trajectory as an example
            #we now have a matrix of shape [100*100,128] which can be fed to the linear layer. The result is the same as 
            #looping 100 times through each timestep examples.
            self.reshapedOut = tf.reshape(self.output, (-1, self.Hidden_units))

            self.linearLayer = tf.matmul(self.reshapedOut, self.W1) + self.B1
            
            #Compute Linear layer and apply dropout
            self.linearLayerDrop = tf.compat.v1.nn.dropout(self.linearLayer, rate = 1 - self.keepProb)

        with tf.compat.v1.variable_scope("Place_Cells_Units"):
            self.W2 = tf.compat.v1.get_variable("Weights_LinearDecoder_placeCells", [self.LinearLayer_units, self.PlaceCells_units], initializer=tf.keras.initializers.GlorotUniform())
            self.B2 = tf.compat.v1.get_variable("Biases_LinearDecoder_placeCells", [self.PlaceCells_units], initializer=tf.keras.initializers.GlorotUniform())
            
            #Compute the predicted Place Cells Distribution
            self.OutputPlaceCellsLayer = tf.matmul(self.linearLayerDrop, self.W2) + self.B2

        with tf.compat.v1.variable_scope("Head_Cells_Units"):
            self.W3 = tf.compat.v1.get_variable("Weights_LinearDecoder_HeadDirectionCells", [self.LinearLayer_units, self.HeadCells_units], initializer=tf.keras.initializers.GlorotUniform())
            self.B3 = tf.compat.v1.get_variable("Biases_LinearDecoder_HeadDirectionCells", [self.HeadCells_units], initializer=tf.keras.initializers.GlorotUniform())   
            
            #Compute the predicted Head-direction Cells Distribution
            self.OutputHeadCellsLayer = tf.matmul(self.linearLayerDrop, self.W3) + self.B3
  
    def buildTraining(self):
        """
        Description: 
            Constructs the training pipeline for the network. Includes:
            - Placeholders for ground-truth distributions of place and head cells across all time steps.
            - Reshaping of output predictions and labels for batch processing.
            - Softmax cross-entropy losses for place and head cells.
            - L2 weight decay for regularization.
            - Gradient computation with RMSProp and manual clipping on decoder weights.
            - Final optimization step that applies the computed gradients.
        -------- Parameters --------
        None
        -------- Returns --------
        None
        """
        #Fed the Ground Truth Place Cells Distribution and Head Direction Cells Distribution
        self.LabelPlaceCells = tf.compat.v1.placeholder(tf.float32, shape=[None, 100, self.PlaceCells_units], name="Labels_Place_Cells")
        self.LabelHeadCells = tf.compat.v1.placeholder(tf.float32,  shape=[None, 100, self.HeadCells_units], name="Labels_Head_Cells")
        
        self.reshapedPlaceCells = tf.reshape(self.LabelPlaceCells, (-1, self.PlaceCells_units))
        self.reshapedHeadCells = tf.reshape(self.LabelHeadCells, (-1, self.HeadCells_units))

        #Compute the errors for each neuron in each trajectory for each timestep [1000,256] and [1000,12] errors
        self.errorPlaceCells = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.reshapedPlaceCells, logits=self.OutputPlaceCellsLayer, name="Error_PlaceCells"))
        self.errorHeadCells = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.reshapedHeadCells, logits=self.OutputHeadCellsLayer, name="Error_HeadCells"))
        #Compute the l2_loss
        l2_loss = (self.weight_decay * tf.compat.v1.nn.l2_loss(self.W3)) + (self.weight_decay * tf.compat.v1.nn.l2_loss(self.W2))

        #Compute mean among truncated errors [10,1] -> [1] (mean error)
        self.meanLoss = self.errorHeadCells + self.errorPlaceCells + l2_loss
        
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate, momentum=0.9)

        self.gvs = self.optimizer.compute_gradients(self.meanLoss)

        #Apply gradient clipping to parameters: Place Cells units (weights, biases) , Head Cells units (weights, biases)
        self.gvs[-4] = [tf.clip_by_value(self.gvs[-4][0], -self.clipping, self.clipping), self.gvs[-4][1]]
        self.gvs[-3] = [tf.clip_by_value(self.gvs[-3][0], -self.clipping, self.clipping), self.gvs[-3][1]]
        self.gvs[-2] = [tf.clip_by_value(self.gvs[-2][0], -self.clipping, self.clipping), self.gvs[-2][1]]
        self.gvs[-1] = [tf.clip_by_value(self.gvs[-1][0], -self.clipping, self.clipping), self.gvs[-1][1]]

        self.opt = self.optimizer.apply_gradients(self.gvs)
    
    def buildTensorBoardStats(self):
        """
        Description: 
            Prepares TensorBoard scalar summaries for monitoring:
            - Mean training loss across batches.
            - Average distance between predicted and actual locations (can be computed externally).
        -------- Parameters --------
        None
        -------- Returns --------
        None
        """
        #Episode data
        self.mn_loss = tf.compat.v1.placeholder(tf.float32)
        self.mergeEpisodeData = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar("mean_loss", self.mn_loss)])

        self.avgD = tf.compat.v1.placeholder(tf.float32)
        self.mergeAccuracyData = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar("average_distance", self.avgD)])
    
    def save_restore_Model(self, restore, epoch=0):
        """
        Description: 
            Saves or restores the model from disk. If `restore` is True, it loads the model parameters from 
            a saved checkpoint. Otherwise, it updates the epoch counter and saves the current model state.
        -------- Parameters --------
        @param restore: Boolean flag indicating whether to restore (True) or save (False) the model.
        @param epoch: Integer epoch value to assign to the model when saving.
        -------- Returns --------
        None
        """
        if restore:
            self.saver.restore(self.sess, "agentBackup/graph.ckpt")
        else:
            self.sess.run(self.epoch.assign(epoch))
            self.saver.save(self.sess, "agentBackup/graph.ckpt")