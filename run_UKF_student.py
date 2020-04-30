"""
Author: Jane Watts, Tim Player, Alex Moody
Email: apham@g.hmc.edu
Date of Creation: 4/23/20
Description:
    Unscented Kalman Filter implementation for filtering true wind estimate
    This  is for sailors or anyone who identifies as a saltyseaperson.

                                  |
                                  |
                                  |
                          |       |
                          |      ---
                         ---     '-'
                         '-'  ____|_____
                      ____|__/    |    /
                     /    | /     |   /
                    /     |(      |  (
                   (      | \     |   \
                    \     |  \____|____\   /|
                    /\____|___`---.----` .' |
                .-'/      |  \    |__.--'    \
              .'/ (       |   \   |.          \
           _ /_/   \      |    \  | `.         \
            `-.'    \.--._|.---`  |   `-._______\
               ``-.-------'-------'------------/
                   `'._______________________.' 

                   Are you to ready set sail, sea people? -Jane
                   Sea people? I hardly sea any people at all during the quarantine! -Tim
                   Quarantine? More like brigantine! -Alex
"""
from __future__ import print_function
import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
from scipy import integrate
import sciplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb


DT = 1.03
EARTH_RADIUS = 6.3781E6  # meters
MAST_HEIGHT = 10
MS_TO_KNOTS = 1 #get correct value for this
lmda = 1 #parameters to stretch or condense sigma points
alpha = 0.7
beta = 2


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
    sigma_points (np.array)  -- sigma points based off of previous estimate
    u_t (np.array)           -- the current control input

    Returns:
    sigma_points_pred (np.array)   -- the predicted state matrix
    """
    #update state with input
    (n,m)= sigma_points.shape
    #reshape matrix for operatiosn
    sigma_points.shape = (n,1,m)
    #initialize predicted sigma points
    sigma_points_pred = np.zeros((n,m))
    #motion model
    for i in range(m): 
        sigma_points_pred[0,i] = u_t[0,0]
        sigma_points_pred[1,i] = u_t[1,0]
        sigma_points_pred[2,i] = wrap_to_pi(u_t[0,0]-sigma_points[0,0,i])/DT
        sigma_points_pred[3,i] = wrap_to_pi(u_t[1,0]-sigma_points[1,0,i])/DT
        sigma_points_pred[4,i] = u_t[2,0]
        sigma_points_pred[5,i] = u_t[3,0]
        sigma_points_pred[6,i] = sigma_points[6,0,i]
        sigma_points_pred[7,i] = sigma_points[7,0,i]
    return sigma_points_pred

"""
                  .
                .'|     .8
               .  |    .8:
              .   |   .8;:        .8
             .    |  .8;;:    |  .8;
            .     n .8;;;:    | .8;;;
           .      M.8;;;;;:   |,8;;;;;
          .    .,"n8;;;;;;:   |8;;;;;;
         .   .',  n;;;;;;;:   M;;;;;;;;
        .  ,' ,   n;;;;;;;;:  n;;;;;;;;;
       . ,'  ,    N;;;;;;;;:  n;;;;;;;;;
      . '   ,     N;;;;;;;;;: N;;;;;;;;;;
     .,'   .      N;;;;;;;;;: N;;;;;;;;;;
    ..    ,       N6666666666 N6666666666
    I    ,        M           M
   ---nnnnn_______M___________M______mmnnn
         "-.                          /
  __________"-_______________________/_________
"""
def motion_uncertainty(sigma_points_pred,mean_bar_t):
    """calculate the covariance matrix resulting from the prediction step

    Parameters:
    mean_t_prev (np.array)          -- the previous state estimate
    sigma_points_pred (np.array)    -- predicted state matrix

    Returns:
    sigma_x_bar_t (np.array)        -- predicted covariance matrix
    """
    #get input array shape
    (n,m) = sigma_points_pred.shape
    #shape array for matrix operations
    sigma_points_pred.shape = (n,1,m)
    #motion model noise
    R_t = .001*np.identity(n) #update with correct value
    #initialize output array
    sigma_x_bar_t = np.zeros((n,n))
    #calculate weights
    weights = np.zeros(m)
    weights[0] = lmda/(n+lmda) + (1-(alpha**2)+beta)
    weights[1:] = 1/(2*(n+lmda))
    #loop through sigma points and add error (using loop bc wrap to pi issues)
    for i in range(m):
        sigma_points_error = np.subtract(sigma_points_pred[:,:,i],mean_bar_t)
        sigma_points_error[0,0] = wrap_to_pi(sigma_points_error[0,0])
        sigma_points_error[1,0] = wrap_to_pi(sigma_points_error[1,0])
        sigma_points_error[4,0] = wrap_to_pi(sigma_points_error[4,0])
        sigma_points_error[6,0] = wrap_to_pi(sigma_points_error[6,0])

        squared_error = np.matmul(sigma_points_error,sigma_points_error.T)
        sigma_x_bar_t = np.add(sigma_x_bar_t,weights[i]*squared_error)
        
    sigma_x_bar_t = np.add(sigma_x_bar_t,R_t) #confirm R_t is not included in summation
    
    return sigma_x_bar_t


def motion_regroup(sigma_points_pred):
    """regroups the predicted sigma points using a weighted average

    Parameters:
    sigma_points_pred (np.array)     -- predicted sigma points

    Returns:
    mean_bar_t (np.array)            -- predicted state estimate
    """

    #get input matrix shape
    (n,m)= sigma_points_pred.shape
    #reshape for matrix operations
    sigma_points_pred.shape = (n,1,m)
    #initialize output array
    mean_bar_t = np.zeros((n,1))
    #calc weights
    weights = np.zeros(m)
    weights[0] = lmda/(n+lmda)
    weights[1:] = 1/(2*(n+lmda))
    #regrou equation
    for i in range(m):
        mean_bar_t = np.add(mean_bar_t,weights[i]*sigma_points_pred[:,:,i])
        mean_bar_t[0,0] = wrap_to_pi(mean_bar_t[0,0])
        mean_bar_t[1,0] = wrap_to_pi(mean_bar_t[1,0])
        mean_bar_t[4,0] = wrap_to_pi(mean_bar_t[4,0])
        mean_bar_t[6,0] = wrap_to_pi(mean_bar_t[6,0])

    #reshape sigma points to match outside dimensions
    sigma_points_pred.shape = (n,m)
    return mean_bar_t

def calc_sigma_points(mean_t_prev, sigma_t_prev):
    """Calculate sigma points to use in prediction and correction step

    Parameters:
    mean_t_prev (np.array)     -- the previous state estimate
    sigma_t_prev (np.array)          -- the previous covariance matrix

    Returns:
    sigma_points (np.array)        -- state matrix
    """
    (n,m) = np.shape(mean_t_prev)

    #how dow we deal with negative covariances in sqrt?
    sigma_points = np.zeros((n,2*n +1))
    sigma_points[:,0] =mean_t_prev[:,0]
    sigma_points[:,1:n+1] = mean_t_prev + np.linalg.cholesky((n+lmda)*sigma_t_prev)[:,0:n] #do we need to wrap sigma values?
    sigma_points[:,n+1:] = mean_t_prev - np.linalg.cholesky((n+lmda)*sigma_t_prev)[:,0:n]
    sigma_points[0,1:] = [wrap_to_pi(x) for x in sigma_points[0,1:]]
    sigma_points[1,1:] = [wrap_to_pi(x) for x in sigma_points[1,1:]]
    sigma_points[4,1:] = [wrap_to_pi(x) for x in sigma_points[4,1:]]
    sigma_points[6,1:] = [wrap_to_pi(x) for x in sigma_points[6,1:]]

    print('big guy', sigma_points[:,0])
    return sigma_points


def prediction_step(mean_t_prev, u_t, sigma_t_prev):
    """Compute the prediction of UKF

    Parameters:
    mean_t_prev (np.array)              -- the previous state estimate
    u_t (np.array)                      -- the control input
    sigma_t_prev (np.array)             -- the previous variance estimate

    Returns:
    mean_bar_t (np.array)               -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)            -- the predicted variance estimate of time t
    sigma_points_pred_final (np.array)  -- the predicted sigma points
    """

    sigma_points = calc_sigma_points(mean_t_prev, sigma_t_prev)
    sigma_points_pred = propogate_state(sigma_points,u_t)
    mean_bar_t = motion_regroup(sigma_points_pred)
    sigma_x_bar_t = motion_uncertainty(sigma_points_pred,mean_bar_t)
    sigma_points_pred_final = calc_sigma_points(mean_bar_t,sigma_x_bar_t)
    
    

    return mean_bar_t, sigma_x_bar_t, sigma_points_pred_final
"""
                                       ..
                                     .(  )`-._
                                   .'  ||     `._
                                 .'    ||        `.
                              .'       ||          `._
                            .'        _||_            `-.
                         .'          |====|              `..
                       .'             \__/               (  )
                     ( )               ||          _      ||
                     /|\               ||       .-` \     ||
                   .' | '              ||   _.-'    |     ||
                  /   |\ \             || .'   `.__.'     ||   _.-..
                .'   /| `.            _.-'   _.-'       _.-.`-'`._`.`
                \  .' |  |        .-.`    `./      _.-`.    `._.-'
                 |.   |  `.   _.-'   `.   .'     .'  `._.`---`
                .'    |   |  :   `._..-'.'        `._..'  ||
               /      |   \  `-._.'    ||                 ||
              |     .'|`.  |           ||_.--.-._         ||
              '    /  |  \ \       __.--'\    `. :        ||
               \  .'  |   \|   ..-'   \   `._-._.'        ||
`.._            |/    |    `.  \  \    `._.-              ||
    `-.._       /     |      \  `-.'_.--'                 ||
         `-.._.'      |      |        | |         _ _ _  _'_ _ _ _ _
              `-.._   |      \        | |        |_|_|_'|_|_|_|_|_|_|
                  [`--^-..._.'        | |       /....../|  __   __  |
                   \`---.._|`--.._    | |      /....../ | |__| |__| |
                    \__  _ `-.._| `-._|_|_ _ _/_ _ _ /  | |__| |__| |
                     \   _o_   _`-._|_|_|_|_|_|_|_|_/   '-----------/
                      \_`.|.'  _  - .--.--.--.--.--.`--------------'
      .```-._ ``-.._   \__   _    _ '--'--'--'--'--'  - _ - _  __/
 .`-.```-._ ``-..__``.- `.      _     -  _  _  _ -    _-   _  __/(.``-._
 _.-` ``--..  ..    _.-` ``--..  .. .._ _. __ __ _ __ ..--.._ / .( _..``
`.-._  `._  `- `-._  .`-.```-._ ``-..__``.-  -._--.__---._--..-._`...```
   _.-` ``--..  ..  `.-._  `._  `- `-._ .-_. ._.- -._ --.._`` _.-`LGB`-.
"""

def calc_meas_sigma_points(sigma_points_pred_final):
    """Calculates predicted measurements for each sigma point

    Parameters:
    sigma_points_pred_final (np.array)  -- the predicted state matrix

    Returns:
    Z_bar_t (np.array)                  -- measurement matrix (2x17)
    """
    
    (n,m)= sigma_points_pred_final.shape
    #initialize output array
    Z_bar_t = np.zeros((2,m))
    for i in range(m): 
        Z_bar_t[:,i] = calc_meas_prediction(sigma_points_pred_final[:,i])
 

    return Z_bar_t


def calc_kalman_gain(sigma_bar_xzt,S_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_bar_xzt (np.array)  -- the cross covariance matrix
    S_t (np.array)            -- the measurement uncertainty

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    K_t = np.matmul(sigma_bar_xzt, np.linalg.inv(S_t))

    return K_t

def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    z_bar_t defined as [z_xLL, z_yLL]
          ,_
          I~
          |\
         /|.\
        / || \
      ,'  |'  \
   .-'.-==|/_--'
   `--'-------'    _ seal _
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
    app_wind_right_of_vane  = app_wind_stb * np.cos(roll) #+ MAST_HEIGHT * roll_dot * MS_TO_KNOTS
    app_wind_fwd_of_vane    = app_wind_fwd

    # take the angle and magnitude of the relative wind vector in the vane frame
    z_AWA = np.arctan2(app_wind_fwd_of_vane, app_wind_right_of_vane)
    z_AWS = np.linalg.norm([app_wind_right_of_vane, app_wind_fwd_of_vane])

    z_bar_t = np.array([z_AWA, z_AWS])

    return z_bar_t

def meas_regroup(Z_bar_t):
    """Regroup the measurement matrix using a weighted average

    Parameters:
    Z_bar_t (np.array)     -- measurement matrix

    Returns:
    z_bar_t (np.array)     -- weighted average of measurement matrix columns
    """
    (n,m)= Z_bar_t.shape
    #reshape Z_bar_t for matrix operations
    Z_bar_t.shape = (n,1,m)
    #initialize output array
    z_bar_t = np.zeros((n,1))
    #calc weights
    weights = np.zeros(m)
    weights[0] = lmda/(n+lmda)
    weights[1:] = 1/(2*(n+lmda))

    for i in range(m):
        z_bar_t = z_bar_t + weights[i]*Z_bar_t[:,:,i]
        z_bar_t[0] = wrap_to_pi(z_bar_t[0])

    #reshape to match expected output dimensions
    Z_bar_t.shape = (n,m)
    return z_bar_t

def meas_uncertainty(sigma_points_pred_final,Z_bar_t,z_bar_t, mean_bar_t):
    """Calculate measurement uncertainty to use in kalman gain

    Parameters:
    sigma_ponts_pred_final (np.array) -- predicted state matrix
    Z_bar_t (np.array)                -- measurement matrix
    z_bar_t (np.array)                -- grouped predicted measurement
    mean_bar_t (np.array)             -- predicted state estimate

    Returns:
    S_t (np.array)                    -- predicted emasurement uncertainty
    sigma_bar_xzt (np.array)          -- cross covariance matrix
    """
    #get shape of each input matrix
    (n,m) = sigma_points_pred_final.shape
    (nz,mz) = Z_bar_t.shape

    #reshape so they have a single column for matrix operations
    sigma_points_pred_final.shape = (n,1,m)
    Z_bar_t.shape = (nz,1,mz)

    #harcode sigma z
    sigma_z_t = 0.000001*np.identity(nz)
    #sigma_z_t = np.zeros((nz,nz))

    #initialize output matrices
    sigma_bar_xzt = np.zeros((n,nz))
    S_t = np.zeros((nz,nz))

    #calc weights
    weights = np.zeros(m)
    weights[0] = lmda/(n+lmda) + (1-alpha**2+beta)
    weights[1:] = 1/(2*(n+lmda))

    
    for i in range(m):
        #calc errors in measurement points and sigma points/wrap to pi
        z_error = (Z_bar_t[:,:,i]-z_bar_t)
        z_error[0,0] = wrap_to_pi(z_error[0,0])
        sigma_points_error = (sigma_points_pred_final[:,:,i]-mean_bar_t)
        sigma_points_error[0,0] = wrap_to_pi(sigma_points_error[0,0])
        sigma_points_error[1,0] = wrap_to_pi(sigma_points_error[1,0])
        sigma_points_error[4,0] = wrap_to_pi(sigma_points_error[4,0])
        sigma_points_error[6,0] = wrap_to_pi(sigma_points_error[6,0])

        #calc uncertainties
        z_squared_error = np.matmul(sigma_points_error,z_error.T)
        sigma_bar_xzt = np.add(sigma_bar_xzt,weights[i]*z_squared_error) #pretty sure dont need wrap to pi but will revisit later
        s_squared_error = np.matmul(z_error,z_error.T)
        S_t = np.add(S_t,weights[i]*s_squared_error)
    
    S_t = np.add(S_t,sigma_z_t) #am unsure if this is summed every sigma point or not
   

    return S_t, sigma_bar_xzt

def correction_step(mean_bar_t, z_t, sigma_x_bar_t, sigma_points_pred_final):
    """Compute the correction of UKF

    Parameters:
    mean_bar_t       (np.array)         -- the predicted state estimate of time t
    z_t           (np.array)            -- the measured state of time t
    sigma_x_bar_t (np.array)            -- the predicted variance of time t
    sigma_points_pred_final (np.array)  -- predicted sigma points

    Returns:
    mean_est_t       (np.array)         -- the filtered state estimate of time t
    sigma_x_est_t (np.array)            -- the filtered variance estimate of time t
    """

    Z_bar_t = calc_meas_sigma_points(sigma_points_pred_final)
    z_bar_t = meas_regroup(Z_bar_t)
    S_t,sigma_bar_xzt = meas_uncertainty(sigma_points_pred_final,Z_bar_t,z_bar_t, mean_bar_t)
    K_t = calc_kalman_gain(sigma_bar_xzt,S_t)
    meas_diff = (z_t-z_bar_t)
    meas_diff[0,0] = wrap_to_pi(meas_diff[0,0])
    
    mean_est_t = mean_bar_t + np.matmul(K_t,meas_diff)
    mean_est_t[0,0] = wrap_to_pi(mean_est_t[0,0])
    mean_est_t[1,0] = wrap_to_pi(mean_est_t[1,0])
    mean_est_t[4,0] = wrap_to_pi(mean_est_t[4,0])
    mean_est_t[6,0] = wrap_to_pi(mean_est_t[6,0])
    K_dot_S = np.matmul(K_t,S_t)
    sigma_x_est_t = sigma_x_bar_t - np.matmul(K_dot_S,K_t.T)

    return mean_est_t, sigma_x_est_t

def plot_graphs(time_stamps, state_estimates, z_AWA, z_AWS, data_TWA, data_TWS):
    plt.figure(1)
    plt.plot(time_stamps, state_estimates[6,:])
    plt.xlabel('Time (s)')
    plt.ylabel('TWA (rad from east)')
    plt.title('Estimated TWA vs Time')

    plt.figure(2)
    plt.plot(time_stamps, state_estimates[7,:])
    plt.xlabel('Time (s)')
    plt.ylabel('TWS kts')
    plt.title('Estimated TWS vs Time')

    plt.figure(3)
    plt.plot(time_stamps, state_estimates[2,:])
    plt.xlabel('Time (s)')
    plt.ylabel('\dot{Roll} (rad/s)')
    plt.title('Roll rate vs Time')

    plt.figure(4)
    plt.plot(time_stamps, z_AWA)
    plt.xlabel('Time (s)')
    plt.ylabel('AWA (rad from starboard)')
    plt.title('Measured AWA vs Time')

    plt.figure(5)
    plt.plot(time_stamps, data_TWA)
    plt.xlabel('Time (s)')
    plt.ylabel('TWA (rad from east)')
    plt.title('Professional TWA vs Time')

    plt.figure(6)
    plt.plot(time_stamps, data_TWS)
    plt.xlabel('Time (s)')
    plt.ylabel('TWS (kts)')
    plt.title('Professional TWS vs Time')

    plt.show()


def main():
    """Run a UKF on logged data from a
                 ,
              |"-,_
              I--(_
             ,I?8,
             d|`888.
            d8| 8888b
           ,88| ?8888b
          ,888| `88888b
         ,8888|  8888g8b
        ,88888|  888PX?8b
       ,888888|  8888bd88,
      o8888888| ,888888888
     d8888888P| d888888888b
  _.d888gggg8'| 8gg88888888,
 '\==-,,,,,,,,|/;,,,,,-==;7
 _ \__...____...__    __/ _ seal _
   ~              ~~~~  ~~
   """

    filepath = "./"
    filename = "2019Aug10_revised"
    data = load_data(filepath + filename)

    # Load data into variables
    time_stamps = data["Timestamp"]
    lat_gps = data["Lat"]
    lon_gps = data["Lon"]
    u_roll  = [wrap_to_pi(x * np.pi/180) for x in data['Heel']]         # ref mast, to the right side of the boat
    u_yaw   = [wrap_to_pi(np.pi/2 - x*np.pi/180) for x in data['HDG']]  # HDG is changed into ref E CCW, initially imported w ref N CW 
    u_v_ang = [wrap_to_pi(np.pi/2 - x*np.pi/180) for x in data["COG"]]   # COG is changed into ref E CCW, initially imported w ref N CW
    u_v_mag = data["SOG"]          
    z_AWA = [wrap_to_pi((np.pi/2) - x*np.pi/180) for x in data["AWA"]]    # AWA is changed into ref E CCW, initially imported w ref N CW
    z_AWS = data["AWS"]
    data_TWD = [wrap_to_pi(np.pi/2 - x*np.pi/180) for x in data["TWD"]] # TWA is changed into ref E CCW, initially imported w ref N CW
    data_TWS = data["TWS"]
    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]
    pdb.set_trace()

    #  Initialize filter
    N = 8  # number of states
    state_est_t_prev = np.array([[u_roll[0],u_yaw[0],wrap_to_pi(u_roll[1]-u_roll[0])/DT,wrap_to_pi(u_yaw[1]-u_yaw[0])/DT,u_v_ang[0],u_v_mag[0],data_TWD[0],data_TWS[0]]]).T #initial state assum global (0,0) is at northwest corner
    var_est_t_prev = 0.01*np.identity(N)

    state_estimates = np.zeros((N,1, len(time_stamps)))
    covariance_estimates = np.zeros((N, N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))
    state_estimates[:,:,0] = state_est_t_prev
    covariance_estimates[:,:,0] = var_est_t_prev

    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        # Get control input
        u_t = np.array([[u_roll[t],u_yaw[t],u_v_ang[t],u_v_mag[t]]]).T

        # Prediction Step
        state_pred_t, var_pred_t, sigma_points_pred= prediction_step(state_est_t_prev,u_t,var_est_t_prev)

        # Get measurement
        z_t = np.array([[z_AWA[t],z_AWS[t]]]).T
        #Correction Step
        state_est_t, var_est_t = correction_step(state_pred_t,z_t,var_pred_t,sigma_points_pred)
        
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data
        state_estimates[:,:, t] = state_est_t
        covariance_estimates[:, :, t] = var_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])
    state_estimates.shape = (N,len(time_stamps))
    #plotting
    plot_graphs(time_stamps, state_estimates, z_AWA, z_AWS, data_TWD, data_TWS)
    # plt.plot(state_estimates[6,0,:])
    # plt.show()
    return 0


if __name__ == "__main__":
    print = sciplot.print_wrapper(print)
    main()
