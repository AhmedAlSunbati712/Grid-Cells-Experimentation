import numpy as np
import matplotlib.pyplot as plt
import pickle
from ratSimulator import RatSimulator

class dataGenerator():
    def __init__(self, number_steps, num_features, pc_units, hd_units):
        #HYPERPARAMETERS
        self.number_steps = number_steps #number of time steps
        self.num_features = num_features #number of features (3 in our case - velocity, sine and cosine of angular velocity)
        self.placeCell_units = pc_units #number of place cell units
        self.headCell_units = hd_units #number of head cell units

        self.ratSimulator=RatSimulator(self.number_steps) #initialise rat simulator

        
    def generateData(self, num_trajectories):
        """
        Description:
            Produces data: 
                - input_data: A (num_traj, num_steps, 3) array where each trajectory, for each
                  timestep is a 3D vector (velocity, sin(angular_velocity), cos(angular_velocity)). 
                - positions: A (num_traj, num_steps, 2) array of the coordinates of position for each timestep
                             of each trajectory.
                - angles: A (num_traj, num_steps, 1) array of the values of head angles for each timestep of each
                          trajectory
        --------- Parameters ---------
        @param num_trajcetories: Number of trajectories to generate
        --------- Returns ---------
        See Description
        """
        input_data = np.zeros((num_trajectories, self.number_steps, 3)) #create list to store input data

        # lists for trajectory data
        velocities = np.zeros((num_trajectories, self.number_steps)) 
        ang_velocities = np.zeros((num_trajectories, self.number_steps))
        angles = np.zeros((num_trajectories, self.number_steps))
        positions = np.zeros((num_trajectories, self.number_steps,2))

        print(">>Generating trajectories")
        for i in range(num_trajectories): #create as many trajectories as batch size
            vel, angVel, pos, angle = self.ratSimulator.generateTrajectory() #get trajectory data from rat simulator

            #store data of ith trajectory
            velocities[i] = vel
            ang_velocities[i] = angVel
            angles[i] = angle
            positions[i] = pos

        #format input data 
        for t in range(self.number_steps):
            input_data[:, t, 0] = velocities[:, t]
            input_data[:, t, 1] = np.sin(ang_velocities[:, t])
            input_data[:, t, 2] = np.cos(ang_velocities[:, t])

        return input_data, positions, angles #return input data and positions and angles (for supervision)
        

    def computePlaceCellsDistrib(self, positions, cell_centers):
        """
        Description:
            Given positions and place cell centers, we calculate place cell distributions for all
            the positions. Formula for c_i (the ith place cell) for a given position vector x is:

            c_i = (exp(-(x-cell_centers[i])^2/2(sigma)^2))/(sum_j^N(exp(-(x-cell_centers[j])^2/2(sigma)^2)))
        --------- Parameters ---------
        @param positions: A (batch_size or num_traj, 2) array of 2D points chosen on trajectories or batch.
        @param cell_centers: The coordinates of the cell centers (means) to calculate the distributions around.
        --------- Returns ---------
        c_normalized (np.ndarray): The normalized place cell distributions for the given positions.
        """
        num_cells = cell_centers.shape[0] #number of place cells
        batch_size = positions.shape[0] #batch size or number of trajectories
        #Place Cell scale
        sigma = 0.01

        sums = np.zeros(batch_size)
        #Every row stores the distribution for a trajectory
        c_unnormalized = np.zeros((batch_size, num_cells))
        #We have 256 elements in the Place Cell Distribution. For each of them
        for i in range(num_cells):
            #compute the sum of all Gaussians
            l2Norms_i = np.sum((positions - cell_centers[i])**2, axis=1)
            c_i = np.exp(-(l2Norms_i/(2*sigma**2)))

            c_unnormalized[:, i] = c_i #store ith Gaussian in distribution corresponding to ith place cell
            sums += c_i

        c_normalized = c_unnormalized/sums[:,None] # normalise the distribution of place cells
        # This returns a (batch_size x num_cells) matrix where (i,j) corresponds to probability of jth place cell in ith trajectory
        return c_normalized 

    def computeHeadCellsDistrib(self,facingAngles, cell_centers):
        """
        Description:
            Given positions and place cell centers, we calculate place cell distributions for all
            the positions. Formula for c_i (the ith place cell) for a given position vector x is:

            c_i = (exp(k * cos(phi - cell_centers[i]))/(sum_j^M(exp(k * cos(phi - cell_centers[j])))
        --------- Parameters ---------
        @param positions: A (batch_size or num_traj, 2) array of 2D points chosen on trajectories or batch.
        @param cell_centers: The coordinates of the cell centers (means) to calculate the distributions around.
        --------- Returns ---------
        h_normalized (np.ndarray): The normalized place cell distributions for the given positions.
        """
        num_cells = cell_centers.shape[0] #number of head cells
        batch_size = facingAngles.shape[0] #batch size or number of trajectories
        #Concentration parameter 
        k = 20

        sums = np.zeros(batch_size)
        #Every row stores the distribution for a trajectory
        h_unnormalized = np.zeros((batch_size,num_cells))
        #We have 12 elements in the Head Direction Cell Distribution. For each of them
        for i in range(num_cells):
            #compute the distribution of head cells
            h_i = np.squeeze(np.exp(k*np.cos(facingAngles - cell_centers[i])))
            h_unnormalized[:,i] = h_i #store ith term in distribution corresponding to ith head cell
            sums += h_i
        
        h_normalized = h_unnormalized/sums[:,None] #normalise the distribution of head cells
        #This returns a (batch_size x num_cells) matrix where (i,j) corresponds to probability of jth place cell in ith trajectory
        return h_normalized