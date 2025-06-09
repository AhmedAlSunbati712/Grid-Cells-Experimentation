import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class Trainer():
    def __init__(self, agent, pc_units, num_steps):
        """
        Description:
            Initializes the Trainer instance with the agent to be trained, the number of place cell units,
            and the total number of timesteps in a trajectory.

        --------- Parameters ---------
        @param agent: The neural network model (usually an LSTM) that outputs place and head cell activity.
        @param pc_units: The number of place cell units (used to split the output labels).
        @param num_steps: The number of time steps in each trajectory used for training/testing.
        
        --------- Returns ---------
        None
        """
        self.agent = agent
        self.pc_units = pc_units
        self.num_steps = num_steps

    def training(self, X, Y, epoch):
        """
        Description:
            Trains the agent for one epoch using truncated backpropagation through time (TBPTT) with a window size of 100.
            Computes and logs average loss across the sequence to TensorBoard.

        --------- Parameters ---------
        @param X: Input tensor of shape (batch_size, num_steps, input_dim), e.g., initial states or observations.
        @param Y: Label tensor of shape (batch_size, num_steps, output_dim), containing true place and head cell activations.
        @param epoch: The current epoch number, used to log the summary to TensorBoard.

        --------- Returns ---------
        None
        """
        # Stores the means of the losses among a training epoch.
        # Used to show the stats on tensorboard
        mn_loss = 0

        #Divide the sequence in 100 steps in order to apply TBTT of 100 timesteps.
        for startB in range(0, self.num_steps, 100):
            endB = startB + 100

            #Retrieve the inputs for the 100 timesteps
            xBatch = X[:, startB:endB]
            
            #Retrieve the labels for the 100 timesteps
            yBatch = Y[:, startB:endB]

            #Retrieve label at timestep 0 for the 100 timesteps
            init_LSTM = yBatch[:,0]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict = { self.agent.X: xBatch, 
                        self.agent.LabelPlaceCells: yBatch[:, :, : self.pc_units],
                        self.agent.LabelHeadCells: yBatch[:, :, self.pc_units : ],
                        self.agent.placeCellGround: init_LSTM[:, :self.pc_units], 
                        self.agent.headCellGround: init_LSTM[:, self.pc_units:],
                        self.agent.keepProb: 0.5}
            
            _, meanLoss, HeadLoss, PlaceLoss = self.agent.sess.run([self.agent.opt,
                                                                  self.agent.meanLoss,
                                                                  self.agent.errorHeadCells,
                                                                  self.agent.errorPlaceCells], feed_dict=feed_dict)

            mn_loss += meanLoss/(self.num_steps//100)

        #training epoch finished, save the errors for tensorboard
        mergedData = self.agent.sess.run(self.agent.mergeEpisodeData, feed_dict={self.agent.mn_loss: mn_loss})
        
        self.agent.file.add_summary(mergedData, epoch)

    def testing(self, X, init_X, positions_array, pcc, epoch):
        """
        Description:
            Tests the trained agent by generating place cell predictions over the full trajectory and compares them
            to the ground-truth positions. It logs the average Euclidean distance between predictions and real positions.
            Optionally plots predicted and real trajectories every 24 epochs.

        --------- Parameters ---------
        @param X: Input tensor of shape (batch_size, num_steps, input_dim) used to feed into the network.
        @param init_X: Ground-truth initial labels used for initializing the LSTM cell and hidden states at each 100-step chunk.
                      Shape: (batch_size, num_chunks, output_dim).
        @param positions_array: True positions corresponding to each time step. Shape: (batch_size, num_steps, 2).
        @param pcc: Array of place cell centers of shape (num_place_cells, 2) used to decode place cell predictions.
        @param epoch: The current epoch number used for logging and optional visualization.

        --------- Returns ---------
        None
        """
        avgDistance=0

        displayPredTrajectories=np.zeros((10,800,2))

        #Divide the sequence in 100 steps
        for startB in range(0, self.num_steps, 100):
            endB = startB+100

            #Retrieve the inputs for the timestep
            xBatch = X[:,startB:endB]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict = { self.agent.X: xBatch, 
                        self.agent.placeCellGround: init_X[:, (startB//100), :self.pc_units], 
                        self.agent.headCellGround: init_X[:, (startB//100), self.pc_units:],
                        self.agent.keepProb: 1}
            
            placeCellLayer = self.agent.sess.run(self.agent.OutputPlaceCellsLayer, feed_dict=feed_dict)
            
            #retrieve the position in these 100 timesteps
            positions = positions_array[:,startB:endB]
            #Retrieve which cell has been activated. Placecell has shape 1000,256. idx has shape 1000,1
            idx = np.argmax(placeCellLayer, axis=1)
            
            #Retrieve the place cell center of the activated place cell
            predPositions = pcc[idx]

            #Update the predictedTrajectory.png
            if epoch%24 == 0:
                displayPredTrajectories[:, startB:endB]=np.reshape(predPositions, (10,100,2))

            #Compute the distance between truth position and place cell center
            distances = np.sqrt(np.sum((predPositions - np.reshape(positions, (-1,2)))**2, axis=1))
            avgDistance += np.mean(distances)/(self.num_steps//100)
        
        #testing epoch finished, save the accuracy for tensorboard
        mergedData = self.agent.sess.run(self.agent.mergeAccuracyData, feed_dict={self.agent.avgD: avgDistance})
        
        self.agent.file.add_summary(mergedData, epoch)

        #Compare predicted trajectory with real trajectory
        if epoch%24 == 0:
            rows = 3
            cols = 3
            fig = plt.figure(figsize=(40, 40))
            for i in range(rows*cols):
                ax = fig.add_subplot(rows, cols, i+1)
                #plot real trajectory
                plt.plot(positions_array[i,:,0], positions_array[i,:,1], 'b', label="Truth Path")
                #plot predicted trajectory
                plt.plot(displayPredTrajectories[i,:,0], displayPredTrajectories[i,:,1], 'go', label="Predicted Path")
                plt.legend()
                ax.set_xlim(0,2.2)
                ax.set_ylim(0,2.2)

            fig.savefig('predictedTrajectory.png')