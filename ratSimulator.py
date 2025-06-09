import numpy as np 
import matplotlib.pyplot as plt
class RatSimulator():
    def __init__(self, n_steps):
        """
        Description:
            Initializes a RatSimulator instance with the number of timesteps in a trajectory.
            It also sets constants related to time resolution, environment bounds, and motion statistics.

        --------- Parameters ---------
        @param n_steps: The number of timesteps to simulate in the trajectory.
        --------- Returns ---------
        None
        """
        self.number_steps = n_steps #initialise number of time steps
        self.dt = 0.02 #resolution of time steps
        self.max_gap = 2.17 #maximum position before detecting a wall
        self.min_gap = 0.03 #minimum position before detecting a wall

        self.vel_scale = 0.13 #scale of Rayleigh distribution for forward velocity
        self.m_ang_vel = 0 #mean angular velocity
        self.stddev_ang_vel = 330 #standard deviation of angular velocity


    #function to generate a random trajectory
    def generateTrajectory(self):
        """
        Description:
            Simulates a single 2D trajectory of a rat-like agent moving in a square environment.
            The agent starts at a random location and updates its position based on sampled
            linear and angular velocities. If the agent approaches a wall, its direction is adjusted.

        --------- Parameters ---------
        None

        --------- Returns ---------
        velocities (np.ndarray): A (n_steps,) array of linear velocities over time.
        ang_vel (np.ndarray): A (n_steps,) array of angular velocities over time (in radians/sec).
        positions (np.ndarray): A (n_steps, 2) array of (x, y) positions over time.
        angle (np.ndarray): A (n_steps,) array of orientation angles (radians) over time.
        """
        velocities = np.zeros((self.number_steps)) #initialise velocities list
        ang_vel = np.zeros((self.number_steps)) #initialise angular velocities list
        positions = np.zeros((self.number_steps, 2)) #initialise positions list
        angle = np.zeros((self.number_steps)) #initialise facing angles list

        for t in range(self.number_steps): #iterate
            #Initialize the agent randomly in the environment
            if(t == 0):
                pos = np.random.uniform(low = 0, high = 2.2, size=(2)) #sample position from uniform distribution
                face_ang = np.random.uniform(low =-np.pi , high = np.pi) #sample facing angle from uniform distribution
                prev_vel = 0 #set previous velocity = 0

            #Check if the agent is near a wall
            if(self.checkWallAngle(face_ang, pos)): #checkWallAngle is a helper function
                #if True, calculate the direction in which to turn by 90 degrees
                rot_vel = np.deg2rad(np.random.normal(self.m_ang_vel, self.stddev_ang_vel))
                dAngle = self.computeRot(face_ang, pos) + rot_vel*self.dt #compute change in angle
                #Velocity reduction factor = 0.25
                vel = np.squeeze(prev_vel - (prev_vel*0.25))
            #If the agent is not near a wall, randomly sample velocity and ang_velocity
            else:
                #Sampling velocity
                vel = np.random.rayleigh(self.vel_scale) #Rayleigh
                #Sampling angular velocity
                rot_vel = np.deg2rad(np.random.normal(self.m_ang_vel, self.stddev_ang_vel)) #Gaussian
                dAngle = rot_vel * self.dt #change in angle

            #Update the position of the agent
            new_pos = pos + (np.asarray([np.cos(face_ang), np.sin(face_ang)])*vel)*self.dt
            
            #Update the facing angle of the agent
            newface_ang = (face_ang + dAngle)
            #Keep the orientation between -np.pi and np.pi
            if(np.abs(newface_ang)>=(np.pi)):     
                newface_ang = -1 * np.sign(newface_ang) * (np.pi - (np.abs(newface_ang)- np.pi))

            #store quantities in respective lists
            velocities[t] = vel
            ang_vel[t] = rot_vel
            positions[t] = pos
            angle[t] = face_ang
            
            pos = new_pos
            face_ang = newface_ang
            prev_vel = vel
        
        '''
        #USED TO DISPLAY THE TRAJECTORY ONCE FINISHED
        fig=plt.figure(figsize=(12,12))
        ax=fig.add_subplot(111)
        ax.set_title("Trajectory agent")
        ax.plot(positions[:,0], positions[:,1])
        ax.set_xlim(0,2.2)
        ax.set_ylim(0,2.2)
        ax.grid(True)

        plt.show()
        '''
        
        return velocities, ang_vel, positions, angle #return all four lists of the trajectory

    #HELPER FUNCTIONS
    def checkWallAngle(self, ratAng, pos):
        """
        Description:
            Checks if the agent is facing toward or is too close to a wall, based on its position
            and orientation. Used to determine whether a wall-avoidance rotation is needed.

        --------- Parameters ---------
        @param ratAng: The current orientation angle of the agent (in radians).
        @param pos: The current (x, y) position of the agent.

        --------- Returns ---------
        is_near_wall (bool): True if the agent is close to and facing toward a wall; False otherwise.
        """
        #This checks the position of the rat according to the quadrant it's facing towards
        if((0<=ratAng and ratAng<=(np.pi/2)) and np.any(pos>self.max_gap)):
          return True
        elif((ratAng>=(np.pi/2) and ratAng<=np.pi) and (pos[0]<self.min_gap or pos[1]>self.max_gap)):
          return True
        elif((ratAng>=-np.pi and ratAng<=(-np.pi/2)) and np.any(pos<self.min_gap)):
          return True
        elif((ratAng>=(-np.pi/2) and ratAng<=0) and (pos[0]>self.max_gap or pos[1]<self.min_gap)):
          return True
        else:
          return False
    
    def computeRot(self,ang, pos):
        """
        Description:
            Computes the corrective rotation needed for the agent to turn away from a nearby wall.
            This helps keep the agent within bounds by nudging it in the opposite direction.

        --------- Parameters ---------
        @param ang: The agent's current orientation angle (in radians).
        @param pos: The current (x, y) position of the agent.

        --------- Returns ---------
        rot (float): A corrective rotation angle (in radians) to avoid a wall.
        """
        #computes change in angle if agent is near a wall
        rot=0
        if(ang>=0 and ang<=(np.pi/2)):
          if(pos[1]>self.max_gap):
            rot=-ang
          elif(pos[0]>self.max_gap):
            rot=np.pi/2-ang
        elif(ang>=(np.pi/2) and ang<=np.pi):
          if(pos[1]>self.max_gap):
            rot=np.pi-ang
          elif(pos[0]<self.min_gap):
            rot=np.pi/2 -ang
        elif(ang>=-np.pi and ang<=(-np.pi/2)):
          if(pos[1]<self.min_gap):
            rot=-np.pi - ang
          elif(pos[0]<self.min_gap):
            rot=-(ang + np.pi/2)
        else:
          if(pos[1]<self.min_gap):
            rot=-ang
          elif(pos[0]>self.max_gap):
            rot=(-np.pi/2) - ang

        return rot