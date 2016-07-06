import numpy as np
import matplotlib.pyplot as plt


N = 1000
dt = 1.0
velocity = 1.0
initial_position = 0.0
move_var = 1. # variance in movement
measure_var = 400. # variance in measurement
number = 50 #number of moves
time = np.arange(0,number*dt,dt) #total time range
np.random.seed(100)





def move(x):
    #Compute new position of the mass in each time step
    dx = velocity + np.random.randn()*np.sqrt(move_var)
    x += dx * dt
    return x

def sense_position(x):
    #Returns measurement of new position.
    measurement = x + np.random.randn()*np.sqrt(measure_var)
    return measurement

def moveSimulation(number):
    #Returns position and measurements in each time step
    position = [initial_position]
    measurement = [sense_position(initial_position)]

    i = 0
    while i < (len(range(number))-1):
        position.append(move(position[-1]))
        measurement.append(sense_position(position[-1]))
        i += 1
    return position,measurement

position,measurement = moveSimulation(number)




def predict(pos, movement):
    #calculate the prior probability from position and movement in each time step

    return (pos[0] + movement[0], pos[1] + movement[1])


def update(prior, measurement):
    '''
    Update the posterior probability
    :param prior: tuple,contain mean and variance of prior
    :param measurement: tuple, contain mean and variance of measurement
    :return: mean and variance of posterior probability
    '''
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement

    y = z - x        # residual
    K = P / (P + R)  # Kalman gain

    x = x + K*y           # posterior
    P = (P * R) / (P + R) # posterior variance
    return x, P


x = (50., 400.)  # initial position feed to filter, N(50, 400)
movement = (velocity*dt, move_var) #movement in each time step, N(1.0,1.0)

#Main loop
xs, predictions = [], []
xvar = []

for i in measurement:
    prior = predict(x, movement)
    likelihood = (i, measure_var)
    x = update(prior, likelihood)

    predictions.append(prior[0])
    xs.append(x[0])
    xvar.append(x[1])


print 'final estimate:', xs[-1]
print 'actual final position:', position[-1]


plt.plot(time,measurement,'ro',label='measurement')
plt.plot(time,xs,label='filter')
plt.plot(time,position,label='true position')
plt.xlabel('time')
plt.ylabel('distance')
plt.legend(loc='upper left')
plt.show()

plt.title('Residual')
plt.plot(time,np.array(position)-np.array(xs),'ro')
plt.show()




goodness = sum((np.array(xs)-np.array(position))**2/(np.array(xvar)*(len(xs)-1)))
print 'goodness of fit is calculated to be: ', goodness