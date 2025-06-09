import tensorflow as tf
import pickle
import numpy as np
import os 
import argparse

from .dataGenerator import dataGenerator
from .ratSimulator import RatSimulator
from .agent import Network
from .trainer import Trainer
from .showGridCells import showGridCells

def prepare_testing_data(pc_units, hc_units, pc_centers, hc_centers, num_trajectories, num_steps = 800, num_features = 3):
    """
    Description: 
        Loads pre-generated test trajectory data (position and head angle) if available,
        or generates new test data if the file is not found. It then constructs an 
        initial LSTM hidden state tensor that encodes spatial and directional information 
        using place and head cell representations.
    --------- Parameters ---------
    @param pc_units: The number of the place cells units.
    @param hc_units: The number of the head cells units.
    @param pc_centers: Randomly initialized pc_centers across a 2D space.
    @param hc_centers: Randomly initialized head direction values [-pi, pi].
    @param num_trajectories: Number of trajectories to generate in this set.
    @param num_steps: Number of timesteps in the trajectory. Default is 800
    @param num_features: Number of features associated with each timestep. In this
                         case we have 3: Velocity, sin(angularVelocity), and cos(angularVelocity)
    -------- Returns --------
    input_data_test (np.ndarray): The input trajectory data for testing. Shape is (num_trajectories, num_steps, num_features = 3)
    init_LSTMStateTest (np.ndarray): The initial hidden state for the LSTM, combining
                                     place cell and head cell distributions.
    posTest (np.ndarray): The test positions associated with the trajectory.
    """
    # Creating the data generator object
    test_data_generator = dataGenerator(num_steps, num_features, pc_units, hc_units)
    # Checking if data already exists, if so then load it.
    if os.path.isfile("trajectoriesDataTesting.pickle"):
        print("\nLoading test data..")
        file_data = pickle.load(open("trajectoriesDataTesting.pickle","rb"))

        # Input data is (linear velocity, sin(angularVelocity), cos(angularVelocity))
        input_data_test = file_data['X']
        pos_test = file_data['pos']
        angle_test = file_data['angle']

    # If not, create new data.
    else:
        print("\nCreating test data..")
        # Generating the data
        input_data_test, pos_test, angle_test = test_data_generator.generateData(num_trajectories)
        mydict = {"X" : input_data_test, # Shape: num_trajectories * num_steps * num_features
                  "pos" : pos_test, # Shape: num_trajectories * num_steps
                  "angle" : angle_test} # Shape: num_trajectories * num_steps
        # Saving the trajectories for later use
        with open('trajectoriesDataTesting.pickle', 'wb') as f:
            pickle.dump(mydict, f)

    init_LSTMState_test = np.zeros((10, 8, pc_units + hc_units))
    for i in range(8):  
        init_LSTMState_test[:, i, :pc_units] = test_data_generator.computePlaceCellsDistrib(pos_test[:,(i*100)], pc_centers)
        init_LSTMState_test[:, i, pc_units:] = test_data_generator.computeHeadCellsDistrib(angle_test[:,(i*100)], hc_centers)

    return input_data_test, pos_test, init_LSTMState_test

def generate_training_input_and_label(train_data_generator, pc_units, hc_units, pc_centers, hc_centers,
                                      num_trajectories, num_steps):
    """
    Description: 
        Generates input data (velocity, sin(angVelocity), cos(angVelocity)) and corresponding
        label data (ground truth distributions of place cells and head cells) for training.
    --------- Parameters ---------
    @param train_data_generator: The data generator object used for generating training data.
    @param pc_units: The number of the place cells units.
    @param hc_units: The number of the head cells units.
    @param pc_centers: Randomly initialized pc_centers across a 2D space.
    @param hc_centers: Randomly initialized head direction values [-pi, pi].
    @param num_trajectories: Number of trajectories to generate in the training set.
    @param num_steps: Number of timesteps in the trajectory.
    -------- Returns --------
    input_data (np.ndarray)
    label_data (np.ndarray)
    """
    input_data, pos, angle = train_data_generator.generateData(num_trajectories)
    label_data = np.zeros((num_trajectories, num_steps, pc_units + hc_units))
    for t in range(num_steps):
        label_data[:, t, :pc_units] = train_data_generator.computePlaceCellsDistrib(pos[:,t], pc_centers)
        label_data[:, t, pc_units:] = train_data_generator.computeHeadCellsDistrib(angle[:,t], hc_centers)
    return input_data, label_data

def train_agent(agent, pc_units, hc_units, pc_centers, hc_centers, num_trajectories, num_steps,
                num_features, SGDSteps,  batch_size, global_step):
    """
    Description: 
        Trains an agent on simulated trajectories. Saves the model once training
        is done or there's an error during training.
    --------- Parameters ---------
    @param agent: The network to be trained.
    @param pc_units: The number of the place cells units.
    @param hc_units: The number of the head cells units.
    @param pc_centers: Randomly initialized pc_centers across a 2D space.
    @param hc_centers: Randomly initialized head direction values [-pi, pi].
    @param num_trajectories: Number of trajectories to generate in the training set.
    @param num_steps: Number of timesteps in the trajectory.
    @param num_features: Number of features associated with each timestep. In this
                         case we have 3: Velocity, sin(angularVelocity), and cos(angularVelocity)
    @param SGDSteps: The number of steps to do in stochastic gradient descent optimization.
    @param batch_size: The number of trajectories to consider at a time.
    @param global_step: a variable used to track the number of training steps (or updates) that have been performed so far. 
    -------- Returns --------
    trainer (Trainer instance): Needed for testing the agent.
    global_step: The updated value of global_step at the end of training.
    """
    # Initializing the trainer module object
    trainer = Trainer(agent, pc_units, num_steps)

    # Creating a generator object to generate train data
    train_data_generator = dataGenerator(num_steps, num_features, pc_units, hc_units)
    while (global_step < SGDSteps):

        # Create training Data at every step to avoid overfitting
        input_data, label_data = generate_training_input_and_label(train_data_generator, pc_units, hc_units,
                                                                   pc_centers, hc_centers, num_trajectories,
                                                                   num_steps)
        for startB in range(0, num_trajectories, batch_size):
            endB = startB + batch_size
            #return a tensor of shape 10, 800, 3
            batchX = input_data[startB : endB]
            #return a tensor of shape 10, 800, 256+12
            batchY = label_data[startB : endB]

            trainer.training(batchX, batchY, global_step)
            global_step += 8
    return trainer, global_step


def test_agent(trainer, agent, pc_centers, input_data_test, init_LSTMState_test, pos_test, global_step):
    """
    Description: 
        Evaluates the trained agent's performance on test trajectories. This involves
        running a forward pass through the model using input velocity/angular data,
        and the initial LSTM state constructed from place and head cell activations.
        The function also saves the model's parameters after testing.
    --------- Parameters ---------
    @param trainer: An instance of the Trainer class used to execute the testing routine.
    @param agent: The trained model (instance of Network) to be evaluated and saved.
    @param pc_centers: A (pc_units, 2) array representing place cell centers in 2D space.
    @param input_data_test: A (num_trajectories, num_steps, num_features) array containing
                            the velocity and angular velocity inputs for testing.
    @param init_LSTMState_test: An array representing the initial LSTM hidden state built from
                                place and head cell distributions.
    @param pos_test: An array of true 2D positions used to
                     evaluate prediction accuracy.
    @param global_step: An integer representing the number of training steps completed;
                        used to save the model checkpoint with appropriate naming.
    -------- Returns --------
    None
    """
    print("\n>>Testing the agent")
    trainer.testing(input_data_test, init_LSTMState_test, pos_test, pc_centers, global_step)
    print(">>Global step:", global_step, "Saving the model..\n")
    agent.save_restore_Model(restore=False, epoch = global_step)

