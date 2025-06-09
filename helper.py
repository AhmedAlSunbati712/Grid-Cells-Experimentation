import tensorflow as tf
import pickle
import numpy as np
import os 
import argparse
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataGenerator import dataGenerator
from ratSimulator import RatSimulator
from agent import Network
from trainer import Trainer
from tensorboard.backend.event_processing import event_accumulator


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
    print(input_data.shape)
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
    step = global_step['step']
    # Initializing the trainer module object
    trainer = Trainer(agent, pc_units, num_steps)

    # Creating a generator object to generate train data
    train_data_generator = dataGenerator(num_steps, num_features, pc_units, hc_units)
    while (step < SGDSteps):

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

            trainer.training(batchX, batchY, step)
        if (step%24==0):
            input_data_test, pos_test, init_LSTMState_test = prepare_testing_data(pc_units, hc_units, pc_centers, hc_centers, num_trajectories=10)
            test_agent(trainer, agent, pc_centers, input_data_test, init_LSTMState_test, pos_test, step)
        step += 8
        global_step['step'] = step
        print("Step: ", step)
    return trainer, step


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

def show_activity_map(agent, num_traj, num_steps, num_features, pc_units, hc_units, llu, bins, pc_centers, hc_centers):
    """
    Description: 
        Generates and visualizes activity maps for each unit in the linear layer of a trained navigation agent.
        The function simulates multiple trajectories using a RatSimulator, feeds velocity and angular velocity 
        data to the agent, and records activations of the linear layer over spatial bins. The resulting 
        activity maps are normalized and saved as an image.

    --------- Parameters ---------
    @param agent: The trained model (instance of Network) with TensorFlow session and linear layer.
    @param num_traj: Number of simulated trajectories to generate.
    @param num_steps: Number of timesteps per trajectory.
    @param num_features: Number of input features (e.g., velocity, angular velocity).
    @param pc_units: Number of place cell units used to encode spatial information.
    @param hc_units: Number of head cell units used to encode directional information.
    @param llu: Number of units in the linear layer (i.e., number of neurons whose activity is recorded).
    @param bins: Number of spatial bins per axis to discretize the environment.
    @param pc_centers: A (pc_units, 2) array of place cell centers in 2D space.
    @param hc_centers: A (hc_units, 1) array of preferred head cell angles.

    -------- Returns --------
    @return normMap: A (llu, bins, bins) array of normalized activity maps for each linear layer unit.
    """
    factor = 2.2 / bins
    activityMap = np.zeros((llu, bins, bins))
    counterActivityMap = np.zeros((llu, bins, bins))

    X = np.zeros((num_traj, num_steps, 3))
    positions = np.zeros((num_traj, num_steps, 2))
    angles = np.zeros((num_traj, num_steps, 1))

    env = RatSimulator(num_steps)

    print(">>Generating trajectory")
    for i in range(num_traj):
        vel, angVel, pos, angle = env.generateTrajectory()
        X[i, :, 0] = vel
        X[i, :, 1] = np.sin(angVel)
        X[i, :, 2] = np.cos(angVel)
        positions[i, :] = pos
        angles[i, :] = angle[:, np.newaxis]

    data_generator = dataGenerator(num_steps, num_features, pc_units, hc_units)
    init_X = np.zeros((num_traj, 8, pc_units + hc_units))
    for i in range(8):
        init_X[:, i, :pc_units] = data_generator.computePlaceCellsDistrib(positions[:, i * 100], pc_centers)
        init_X[:, i, pc_units:] = data_generator.computeHeadCellsDistrib(angles[:, i * 100], hc_centers)

    print(">>Computing activity maps")
    batch_size = 500
    for startB in range(0, num_traj, batch_size):
        endB = startB + batch_size
        for startT in range(0, num_steps, 100):
            endT = startT + 100
            xBatch = X[startB:endB, startT:endT]
            feed_dict = {
                agent.X: xBatch,
                agent.placeCellGround: init_X[startB:endB, startT // 100, :pc_units],
                agent.headCellGround: init_X[startB:endB, startT // 100, pc_units:]
            }
            linearNeurons = agent.sess.run(agent.linearLayer, feed_dict=feed_dict)
            posReshaped = np.reshape(positions[startB:endB, startT:endT], (-1, 2))

            for t in range(linearNeurons.shape[0]):
                bin_x, bin_y = (posReshaped[t] // factor).astype(int)
                bin_x = min(bin_x, bins - 1)
                bin_y = min(bin_y, bins - 1)
                activityMap[:, bin_y, bin_x] += np.abs(linearNeurons[t])
                counterActivityMap[:, bin_y, bin_x] += 1

    result = activityMap / np.maximum(counterActivityMap, 1)
    normMap = (result - np.min(result)) / (np.max(result) - np.min(result))

    os.makedirs("activityMaps", exist_ok=True)
    fig = plt.figure(figsize=(80, 80))
    cols, rows = 16, 32
    for i in range(llu):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(normMap[i], cmap="jet", origin="lower")
        plt.axis('off')
    fig.savefig('activityMaps/neurons.jpg')

    return normMap


def show_correlation_map(normMap):
    """
    Description: 
        Computes and visualizes spatial autocorrelation maps (via 2D cross-correlation) for each normalized 
        activity map from the agent's linear layer. Each resulting correlation map captures the periodic 
        structure or symmetry of the neuron's spatial firing pattern and is saved as a composite image.

    --------- Parameters ---------
    @param normMap: A (llu, bins, bins) array of normalized activity maps, typically output from
                    `show_activity_map`.

    -------- Returns --------
    None
    """
    os.makedirs("corrMaps", exist_ok=True)
    llu = normMap.shape[0]
    fig = plt.figure(figsize=(80, 80))
    cols, rows = 16, 32
    for i in range(llu):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(correlate2d(normMap[i], normMap[i]), cmap="jet", origin="lower")
        plt.axis('off')
    fig.savefig('corrMaps/neurons.jpg')

def plot_model_training_history(path_to_tensorboard):
    """
    Description: 
        Loads training logs from TensorBoard event files and plots the evolution of the 'average_distance' 
        metric over time. This helps visualize how well the agent learns to approximate its true trajectory 
        throughout training.

    -------- Parameters --------
    @param path_to_tensorboard: String path to the TensorBoard event log directory or file 
                                (e.g., "tensorboard/events.out.tfevents.*").

    -------- Returns --------
    None
    """
    ea = event_accumulator.EventAccumulator(path_to_tensorboard) 
    ea.Reload()
    timepoints = []
    average_distance_list = []
    for scalar in ea.Scalars('average_distance'):
        timepoints.append(scalar.step)
        average_distance_list.append(scalar.value)

    plt.scatter(timepoints, average_distance_list, label='Average Distance over time')

    plt.xlabel('Training Step')
    plt.ylabel('Average distance from true path')
    plt.title('Average Distance over time')
    plt.legend()

    plt.show()