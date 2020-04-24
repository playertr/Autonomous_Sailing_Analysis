"""
Author: Andrew Q. Pham
Email: apham@g.hmc.edu
Date of Creation: 2/26/20
Description:
    Extended Kalman Filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 3
    Student code version with parts omitted.
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import pdb
from scipy.stats import norm
import scipy

NUM_PARTICLES = 1000

# define the randomness in theta & distance for propagating state
# STDDEV_THETA =  0.0872665 # radians, 5 degrees
STDDEV_THETA =  0.261799 # radians, 15 degrees
STDDEV_DISTANCE = 0.1 # meters
# stddev for defining particle weight in correction step
STDDEV_MEAS_ERR = 0.5 # meters
STDDEV_INIT = 100 # in any dimension, the initial particle array with be randomized with this.

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.1
X_LANDMARK = 5.  # meters
Y_LANDMARK = -5.  # meters
EARTH_RADIUS = 6.3781E6  # meters
DELTA_T = 0.1  # assume "10^5" uS = 0.1 s


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(float(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """

    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin) * \
        math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propagate_state(x_t_prev, u_t):
    """propagate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    state vector is [x_t; y_t; theta_t] all in the global frame

    u_t  (float)         -- the yaw value, our control input 

    """
    x_prev = x_t_prev[0]
    y_prev = x_t_prev[1]
    theta_prev = x_t_prev[2]

    # sample random theta_prev & distance
    samp_theta = theta_prev + np.random.normal(0, STDDEV_THETA)
    samp_dist = np.random.normal(0, STDDEV_DISTANCE)

    # sample predicted state x_t_i with probability P(x_t_i | x_t-1_i , u_t)
    x_t = x_prev + samp_dist * np.cos(samp_theta)
    y_t = y_prev + samp_dist * np.sin(samp_theta)
    
    # reset theta to be the measured yaw value
    theta_t = u_t 
    theta_t = wrap_to_pi(theta_t)

    x_bar_t = np.array([x_t, y_t, theta_t])

    return x_bar_t

def prediction_step(P_t_prev, u_t):
    """Compute the prediction of EKF

    Parameters:
    P_t_prev (np.array)         -- the previous state matrix, [[x, y, theta, w]^T [x, y, theta, w]^T ...]
    u_t (np.array)              -- the control input

    Returns:
    P_t_predict (np.array)      -- the predicted state matrix after propagating all particles forward in time
    """

    # Iterate through every column (particle) of the state matrix and propagate them forward.
    P_t_predict = np.zeros(shape=P_t_prev.shape)

    for i in range(NUM_PARTICLES):
        P_t_predict[:3,i] = propagate_state(P_t_prev[:3,i], u_t) #at this point the weight stays zero

    return P_t_predict

def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    z_bar_t defined as [z_xLL, z_yLL]
    """

    x_t = x_bar_t[0]
    y_t = x_bar_t[1]
    theta_t = x_bar_t[2]
    delta_x = X_LANDMARK - x_t
    delta_y = Y_LANDMARK - y_t

    # Tim has the derivation for this
    z_xLL = np.sin(theta_t) * delta_x - np.cos(theta_t) * delta_y
    z_yLL = np.cos(theta_t) * delta_x + np.sin(theta_t) * delta_y

    z_bar_t = np.array([z_xLL, z_yLL])

    return z_bar_t

def calc_mean_state(P_t):
    """Compute the mean state at time t using the set of particles

    Parameters:
    P_t                  (np.array)    -- the particle state matrix at time t

    Returns:
    state_est_t          (np.array)    -- the averaged state estimate at time t
    """
    # if we expect our particles to diverge into multiple clumps, we could implement a clustering algorithm here!
    # weighted average of multiple numbers: a*x1 + b*x2 + c*x3 / 3
    x_mean = np.average(P_t[0,:], weights=P_t[3,:])
    y_mean = np.average(P_t[1,:], weights=P_t[3,:])
    theta_mean = np.average(P_t[2,:], weights=P_t[3,:])
    state_est_t = np.array([x_mean, y_mean, theta_mean])

    return state_est_t

def correction_step(P_t_predict, z_t):
    """Compute the correction of EKF

    Parameters:
    P_t_predict         (np.array)    -- the predicted state matrix time t
    z_t                 (np.array)    -- the measured state of time t

    Returns:
    P_t                 (np.array)    -- the final state matrix estimate of time t
    """
    normal_object = scipy.stats.norm(0, STDDEV_MEAS_ERR)

    # Calculate weight for each particle j
    for j in range(NUM_PARTICLES):
        # calculate what the particle thinks the measurement should be 
        z_bar_t = calc_meas_prediction(P_t_predict[:3,j])

        # find the geometric distance between z_t and z_pred (Pythagorean theorem)
        distance = np.linalg.norm(z_t - z_bar_t)

        # weight is zero-mean normal PDF evaluated at d (what variance?)
        weight = normal_object.pdf(distance)

        P_t_predict[3,j] = weight

    # Sample from the set of particles proportional to their weights
    # draw another card from the deck of particle-cards with probability w

    weights = P_t_predict[-1,:] / sum(P_t_predict[-1,:])
    indices = np.random.choice(range(NUM_PARTICLES), size=(NUM_PARTICLES), replace=True, p=weights)

    P_t = np.zeros(shape=P_t_predict.shape)
    for m in range(len(indices)):
        P_t[:,m] = P_t_predict[:,indices[m]]
    
    state_est_t = calc_mean_state(P_t)
    return P_t, state_est_t


def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filepath = "./logs/"
    filename = "2020_2_26__16_59_7"
    # filename = "2020_2_26__17_21_59"
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        data = filter_data(data)
        save_data(f_data, filepath+filename+"_filtered.csv")

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    # Tim changed yaw to radians # Tim is a superstar!
    yaw_lidar = [wrap_to_pi(x * -np.pi / 180) for x in data["Yaw"]]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #  Initialize filter
    N = 3  # number of states

    # Randomly generate initial particles from a normal distribution centered around (at 0,0,0)
    # Use small STDDEV_INIT for known start position 
    # Use large STDDEV_INIT for random start position
    P_t_prev = np.random.normal(25,STDDEV_INIT, size=(N+1,NUM_PARTICLES))
    P_t_prev[-1, :] = 1.0 / NUM_PARTICLES # assign equal weights to all paqurticles
    # P_t_prev[-1, :] = P[-1,:] / sum(P[-1,:]) 


    state_estimates = np.empty((N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))
    errorsq = np.empty(len(time_stamps))
    RMSE = np.empty(len(time_stamps))

    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        # Get control input
        u_t = yaw_lidar[t]

        # Prediction Step
        P_t_predict = prediction_step(P_t_prev, u_t)

        # Get measurement
        z_t = np.array([x_lidar[t], y_lidar[t]])

        # Correction Step
        P_t, state_est_t = correction_step(P_t_predict, z_t)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        P_t_prev = P_t

        # Log Data
        state_estimates[:, t] = state_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])

        # RMSE
            # expected_path_x = 0 to 10 to 10 back to 0 to 0
            # expected_path_y = 0 to  0 to -10 to -10 back to 0
            # written for Lab 3
        if (0 < t < 220):
            errorsq[t] = (0 - state_est_t[1])**2
        elif (220 < t < (420 + 6.9)): #b4l2a0z6e9it
            errorsq[t] = (10 - state_est_t[0])**2
            #print(state_est_t[1])
        elif (421 < t < 605):
            errorsq[t] = (-10 - state_est_t[1])**2
        else: # 600 < t < 810
            errorsq[t] = (0 - state_est_t[0])**2
        RMSE[t] = np.sqrt(np.sum(errorsq[0:t]/t))

        # Plot Results
        # Plot Estimated Path & Expected Path & GPS
        # Path tracking error 
        plt.figure(1)
        plt.plot(gps_estimates[0],
                        gps_estimates[1], 'b.', label='GPS (Expected Path)',zorder=40)
        if np.mod(t, 30) == 0:
            plt.figure(1)
            plt.quiver(state_estimates[0, t], state_estimates[1, t], np.cos(
                state_estimates[2, t]), np.sin(state_estimates[2, t]), color='r',label='Estimated State', zorder=50)

            skip_num = 40
            plt.scatter(P_t[0,::skip_num], P_t[1,::skip_num], color='g', label='Particles', s=2, zorder=10)
            plt.xlim(-4, 14)
            plt.ylim(-14, 4)
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            if t == 0:
                expected_path_x = [0, 10, 10, 0, 0]
                expected_path_y = [0, 0, -10, -10, 0]
                plt.plot(expected_path_x,expected_path_y, 'k', label='Perfect Path', zorder=0, linewidth=2)
                plt.plot(X_LANDMARK,Y_LANDMARK, 'mo', label='Landmark Location')
                plt.legend()
            plt.pause(0.0001)

        plt.figure(2)
        plt.plot(t, errorsq[t], 'k.')

        plt.xlabel('Timestep (0.1 s)')
        plt.ylabel('Squared Error (m^2)')
        if t == 0:
            plt.legend()

        print(t)
        plt.pause(0.0001)
    print(RMSE[-1])
    plt.show()
    return 0


if __name__ == "__main__":
    main()
