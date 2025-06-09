from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

ea = event_accumulator.EventAccumulator('./tensorboard') 
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