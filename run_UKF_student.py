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
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.1
X_LANDMARK = 5.  # meters
Y_LANDMARK = -5.  # meters
EARTH_RADIUS = 6.3781E6  # meters
MAST_HEIGHT = 10
MS_TO_KNOTS = 1 #get correct value for this
lmda = 1
alpha = 1
beta = 1


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}

    # Note: we chose the set of data from t = 7000 to 8000 seconds, based on the .mat file.
    header = ["Timestamp","BSP","AWA","AWS","TWA","TWS","TWD","HDG","Heel","Rake","Lat","Lon","COG","SOG"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("data_log.txt", "w")
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

    return data


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
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx-1]]
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
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
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


def propogate_state(sigma_points, u_t):
    """Propogate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    """STUDENT CODE START"""
    
    (n,m)= sigma_points.shape
    sigma_points_pred = np.zeros((n,m))
    for i in range(m): 
        sigma_points_pred[0,i] = u_t[0]
        sigma_points_pred[1,i] = u_t[1]
        sigma_points_pred[2,i] = wrap_to_pi(u_t[0]-sigma_points[0,i])/DT
        sigma_points_pred[3,i] = wrap_to_pi(u_t[1]-sigma_points[1,i])/DT
        sigma_points_pred[4,i] = u_t[2]
        sigma_points_pred[5,i] = u_t[3]
        sigma_points_pred[6,i] = sigma_points[6,i]
        sigma_points_pred[7,i] = sigma_points[7,i]
    sigma_points_pred = sigma_points
    """STUDENT CODE END"""

    return sigma_points_pred


def motion_uncertainty(sigma_points_pred,mean_bar_t,lmda,alpha,beta):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    """STUDENT CODE START"""
    (n,m) = sigma_points_pred.shape
    R_t = np.zeros((n,1))
    sigma_x_bar_t = np.zeros((n,1))
    weights = np.zeros(m)
    weights[0] = lmda/(n+lmda) + (1-alpha**2+beta)
    weights[1:] = 1/(2*(n+lmda))
    for i in range(m):
        squared_error = np.matmul((sigma_points_pred[:,i]-mean_bar_t),(sigma_points_pred[:,i]-mean_bar_t).T) #slightly confused why this is a scalar and not a matrix
        mean_bar_t = mean_bar_t + weights[i]*squared_error+R_t
        mean_bar_t[0] = wrap_to_pi(mean_bar_t[0])
        mean_bar_t[1] = wrap_to_pi(mean_bar_t[1])
        mean_bar_t[4] = wrap_to_pi(mean_bar_t[4])
        mean_bar_t[6] = wrap_to_pi(mean_bar_t[6])

    
    """STUDENT CODE END"""

    return sigma_x_bar_t


def motion_regroup(sigma_points_pred,lmda):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
    u_t (np.array)          -- the current control input

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """

    """STUDENT CODE START"""
    (n,m)= sigma_points_pred.shape
    mean_bar_t = np.zeros((n,1))
    weights = np.zeros(m)
    weights[0] = lmda/(n+lmda)
    weights[1:] = 1/(2*(n+lmda))
    for i in range(m):
        mean_bar_t = mean_bar_t + weights[i]*sigma_points_pred[:,i]
        mean_bar_t[0] = wrap_to_pi(mean_bar_t[0])
        mean_bar_t[1] = wrap_to_pi(mean_bar_t[1])
        mean_bar_t[4] = wrap_to_pi(mean_bar_t[4])
        mean_bar_t[6] = wrap_to_pi(mean_bar_t[6])

    """STUDENT CODE END"""

    return mean_bar_t

def calc_sigma_points(mean_t_prev, sigma_t_prev, lmda):
    """Calculate sigma points to use in prediction and correction step

    Parameters:
    mean_t_prev (np.array)     -- the previous state estimate
    sigma_t_prev (np.array)          -- the previous covariance matrix

    Returns:
    sigma_points (np.array)        -- state matrix
    """
    n,m = np.shape(mean_t_prev)
    sigma_points = np.zeros((n,2*n +1))
    sigma_points[:,0] =mean_t_prev
    sigma_points[:,1:n+1] = mean_t_prev + np.sqrt((n+lmda)*sigma_t_prev)[0:n] #check indexing/wrap to pi
    sigma_points[:,n+1:] = mean_t_prev + np.sqrt((n+lmda)*sigma_t_prev)[0:n]
    return sigma_points


def prediction_step(mean_t_prev, u_t, sigma_t_prev):
    """Compute the prediction of EKF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    sigma_x_t_prev (np.array)   -- the previous variance estimate

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)    -- the predicted variance estimate of time t
    """

    """STUDENT CODE START"""
    sigma_points = calc_sigma_points(mean_t_prev, sigma_t_prev, lmda)
    sigma_points_pred = propogate_state(sigma_points,u_t)
    mean_bar_t = motion_regroup(sigma_points_pred,lmda)
    sigma_x_bar_t = motion_uncertainty(sigma_points_pred,mean_bar_t,lmda,alpha,beta)
    sigma_points_pred_final = calc_sigma_points(mean_bar_t,sigma_x_bar_t,lmda)
    
    """STUDENT CODE END"""

    return [mean_bar_t, sigma_x_bar_t, sigma_points_pred_final]


def calc_meas_sigma_points(sigma_points_pred_final):
    """Calculate the Jacobian of your measurment model with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurment model
    """
    """STUDENT CODE START"""
    (n,m)= sigma_points_pred_final.shape
    Z_bar_t = np.zeros((2,m))
    for i in range(m): 
        Z_bar_t[:,i] = calc_meas_prediction(sigma_points_pred_final[:,i])
    
    """STUDENT CODE END"""

    return Z_bar_t


def calc_kalman_gain(sigma_bar_xzt,S_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    """STUDENT CODE START"""
    K_t = np.matmul(sigma_bar_xzt, np.linalg.inv(S_t))
    """STUDENT CODE END"""

    return K_t

def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    z_bar_t defined as [z_xLL, z_yLL]
    """

    roll      = x_bar_t[0]
    yaw       = x_bar_t[1]
    roll_dot  = x_bar_t[2]
    yaw_dot   = x_bar_t[3]
    v_ang     = x_bar_t[4]
    v_mag     = x_bar_t[5]
    TWA       = x_bar_t[6]
    TWS       = x_bar_t[7]

    # convert true wind from polar coordinates to cartesian
    TWS_x_comp = TWS * np.cos(TWA) # TW East
    TWS_y_comp = TWS * np.sin(TWA) # TW North

    # convert v_boat into cartesian coordinates 
    # note: this is where the wind is coming from
    v_G_x = v_mag * np.cos(v_ang)
    v_G_y = v_mag * np.sin(v_ang)

    # add components to get wind vector relative to boat in global frame
    app_wind_east   = TWS_x_comp + v_G_x
    app_wind_north  = TWS_y_comp + v_G_y
    
    # rotate into boat frame to get wind vector relative to boat in (starboard, forward) frame
    app_wind_stb    = app_wind_east * np.sin(yaw) - app_wind_north * np.cos(yaw)
    app_wind_fwd    = app_wind_east * np.cos(yaw) + app_wind_north * np.sin(yaw)

    # reduce the starboard component by the cosine of the heel (roll) angle
    # to account for out-of-plane measurement of the wind vector
    # note: if we want to account for mast twist later, this is where we do it.
    app_wind_right_of_vane  = app_wind_stb * np.cos(roll) + MAST_HEIGHT * roll_dot * MS_TO_KNOTS
    app_wind_fwd_of_vane    = app_wind_fwd

    # take the angle and magnitude of the relative wind vector in the vane frame
    z_AWA = np.arctan2(app_wind_fwd_of_vane, app_wind_right_of_vane)
    z_AWS = np.linalg.norm([app_wind_right_of_vane, app_wind_fwd_of_vane])

    z_bar_t = np.array([z_AWA, z_AWS])

    return z_bar_t

def meas_regroup(Z_bar_t,lmda):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
    u_t (np.array)          -- the current control input

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """

    """STUDENT CODE START"""
    (n,m)= Z_bar_t.shape
    z_bar_t = np.zeros((n,1))
    weights = np.zeros(m)
    weights[0] = lmda/(n+lmda)
    weights[1:] = 1/(2*(n+lmda))
    for i in range(m):
        z_bar_t = z_bar_t + weights[i]*Z_bar_t[:,i]
        mean_bar_t[0] = wrap_to_pi(mean_bar_t[0])
        mean_bar_t[1] = wrap_to_pi(mean_bar_t[1])
        mean_bar_t[4] = wrap_to_pi(mean_bar_t[4])
        mean_bar_t[6] = wrap_to_pi(mean_bar_t[6])

    """STUDENT CODE END"""

    return z_bar_t

def correction_step(mean_bar_t, z_t, sigma_x_bar_t, sigma_points_pred_final):
    """Compute the correction of EKF

    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    z_t           (np.array)    -- the measured state of time t
    sigma_x_bar_t (np.array)    -- the predicted variance of time t

    Returns:
    x_est_t       (np.array)    -- the filtered state estimate of time t
    sigma_x_est_t (np.array)    -- the filtered variance estimate of time t
    """

    """STUDENT CODE START"""
    Z_bar_t = calc_meas_sigma_points(sigma_points_pred_final)
    z_bar_t = meas_regroup(Z_bar_t,lmda)
    S_t,sigma_bar_xzt = meas_uncertainty(sigma_points_pred_final,Z_bar_t,z_bar_t)
    K_t = calc_kalman_gain(sigma_bar_xzt,S_t)
    mean_t = mean_bar_t + np.matmul(K_t,(z_t-z_bar_t))
    sigma_x_est_t = sigma_x_bar_t - np.mult(K_t,np.matmul(S_t,K_t.T))

    """STUDENT CODE END"""

    return [x_est_t, sigma_x_est_t]


def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filepath = ""
    filename = '2020_2_26__16_59_7_filtered'
    data, is_filtered = load_data(filepath + filename)

    '''
    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        data = filter_data(data)
        save_data(f_data, filepath+filename+"_filtered.csv")
    '''

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    yaw_lidar = data["Yaw"]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    #plt.plot(list(map(wrap_to_pi, [-i*np.pi/180 for i in yaw_lidar])))
    #plt.plot(yaw_lidar)
    #plt.show()

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #  Initialize filter
    """STUDENT CODE START"""
    N = 7 # number of states
    state_est_t_prev = np.array([0,0,0,0,0,0,0]) #initial state assum global (0,0) is at northwest corner
    var_est_t_prev = np.identity(N)

    state_estimates = np.zeros((N, len(time_stamps)))
    covariance_estimates = np.zeros((N, N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))

    state_estimates[:,-1] = state_est_t_prev
    covariance_estimates[:,:,-1] = var_est_t_prev

    #plotting stuff
    theta_diff= []
    """STUDENT CODE END"""

    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        # Get control input
        """STUDENT CODE START"""
        transform = np.array([[np.cos(state_est_t_prev[2]), -np.sin(state_est_t_prev[2]),0],[np.sin(state_est_t_prev[2]), np.cos(state_est_t_prev[2]),0],[0,0,1]])
        u_t = np.array([[x_ddot[t]], [y_ddot[t]],[wrap_to_pi(-yaw_lidar[t]*(np.pi/180))]])
        u_t_global = np.matmul(transform,u_t)
        state_est_t_prev = state_estimates[:,t-1]
        var_est_t_prev = covariance_estimates[:,:,t-1]
        """STUDENT CODE END"""

        # Prediction Step
        state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t_global, var_est_t_prev)

        # Get measurement
        """STUDENT CODE START"""
        z_t = np.array([[x_lidar[t]],[y_lidar[t]],[wrap_to_pi(-yaw_lidar[t]*(np.pi/180))]])
        """STUDENT CODE END"""
        #Correction Step
        state_est_t, var_est_t = correction_step(state_pred_t,
                                                    z_t,
                                                    var_pred_t)
        # if t%2 == 1:    
        #     # Correction Step
        #     state_est_t, var_est_t = correction_step(state_pred_t,
        #                                             z_t,
        #                                             var_pred_t)
        # else:
        #     state_est_t, var_est_t = state_pred_t, var_pred_t

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data
        state_est_t.shape = (7,)
        state_estimates[:, t] = state_est_t
        covariance_estimates[:, :, t] = var_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])

    """STUDENT CODE START"""
    print('donkeh', covariance_estimates[:,:,0])
    #make a square:
    Width = 10
    square = np.array([[0,0]])
    square_x= []
    square_y= [] 
    for i in range(Width):
        square_x.append(0)
        square_x.append(i)
        square_x.append(Width)
        square_x.append(i)
        square_y.append(i)
        square_y.append(0)
        square_y.append(i)
        square_y.append(Width)
        # square = np.concatenate((square,[[0,i]]), axis=0)
        # square =np.concatenate((square,[[i,0]]), axis=0)
        # square = np.concatenate((square,[[Width,i]]), axis=0)
        # square = np.concatenate((square,[[i,Width]]), axis=0)
    squarex = [0,10,10,0,0]
    squarey = [0,0,-10,-10,0]
    plt.plot(state_estimates[0,:],state_estimates[1,:],'rx',label='estimates')
    # print(square)
    plt.plot(squarex,squarey,label='expected path')
    plt.plot(gps_estimates[0,:],gps_estimates[1,:],':',label='GPS Measurements')
    plt.ylabel('y position (m)')
    plt.xlabel('x position (m)')
    plt.legend(loc='best')
    plt.show()

    #state estimate plot
    fig, ax = plt.subplots(1,1)
    ax.plot(state_estimates[0,:],state_estimates[1,:],'r-.',label='estimates')
    ax.plot(gps_estimates[0,:],gps_estimates[1,:],':',label='GPS Measurements')
    ax.plot(squarex,squarey,label='expected path')
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    ax.legend(loc='best')
    e1 = patches.Ellipse((state_estimates[0,400], state_estimates[1,400]), covariance_estimates[0,0,400], covariance_estimates[1,1,400],
                     angle=state_estimates[2,100]+np.pi/2, linewidth=2, fill=False, zorder=2)
    print('first elipse', covariance_estimates[0,0,100], covariance_estimates[1,1,100])
    e2 = patches.Ellipse((state_estimates[0,0], state_estimates[1,0]), covariance_estimates[0,0,0], covariance_estimates[1,1,0],
                     angle=state_estimates[2,0]+np.pi/2, linewidth=2, fill=False, zorder=2)
    ax.add_patch(e2)
    # ax.add_patch(e2)
    plt.show()

    #yaw angle over time
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(state_estimates[2,:]))*DT,state_estimates[2,:])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('yaw angle (rad)')
    plt.show()

    #covariance matrix diagonals over time.
    fig, ax = plt.subplots(3,1)
    ax[0].plot(np.arange(len(covariance_estimates[0,0,:]))*DT,covariance_estimates[0,0,:])
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('X Covariance')
    ax[1].plot(np.arange(len(covariance_estimates[1,1,:]))*DT,covariance_estimates[1,1,:])
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Y Covariance')
    ax[2].plot(np.arange(len(covariance_estimates[2,2,:]))*DT,covariance_estimates[2,2,:])
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('Theta Covariances')
    plt.show()

    #RMS error (not robust)
    error = []
    residuals = []
    for i in range(len(state_estimates[0,:])):
        x = state_estimates[0,i]
        print('yeegdo', x)
        y = state_estimates[1,i]
        if x >= 0 and x <=10:
            if y>-5:
                distance = y**2
            else:
                distance = (-10-y)**2
        elif  x< 0 or x >10:
            if x>5:
                distance = (10-x)**2
            else:
                distance = (x)**2
        residuals.append(distance)
        error.append(np.sqrt(np.mean(residuals)))
    fig,ax = plt.subplots(1,1)
    ax.plot(np.arange(len(error))*DT,error)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('RMS Tracking Error (m)')
    plt.show()

    """STUDENT CODE END"""
    return 0


if __name__ == "__main__":
    main()
