"""
Author: Tim Player, Jane Cho Watts, Alex Moody (aka the Salty Seapeople)
Email: tplayer@hmc.edu, jwatts@hmc.edu, amoody@hmc.edu
Date of Creation: 4/23/20
Description:
    Particle Filter implementation to filtering true wind estimate estimate
    This code is for our Final Project of E205 - Systems Simulation.

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
import pdb
import sciplot
from scipy.stats import norm
import scipy


NUM_PARTICLES   = 500 # reducing this makes it faster, but it may not converge.
MAST_HEIGHT     = 33 # meters
MS_TO_KNOTS     = 1.944 # knots per m/s

STDDEV_DISTANCE = 0.1 # meters

STDDEV_INIT = 20 # in any dimension, the initial particle array with be randomized with this.

# Propagation variance  (1-second timestep)
# 0.0872665 radians = 5 degrees
# 0.261799 radians  = 15 degrees
STDDEV_ROLL         = 0.01745 # radians, 1 deg
STDDEV_YAW          = 2 * 0.01745 # radians, 5 deg
STDDEV_ROLL_DOT     = 0.01 # m/s^2
STDDEV_YAW_DOT      = 0.01 # m/s^2
STDDEV_V_ANG        = 2 * 0.01745 # radians, 5 deg
STDDEV_V_MAG        = 0.5 #m/s
STDDEV_TWD          = 2 * 0.01745 # radians, 1 deg
STDDEV_TWS          = 0.5 # m/s

# Measurement variance
STDDEV_MEAS_AWA     = 2 * 0.01745 # radians, 1 deg
STDDEV_MEAS_AWS     = 0.5 # m/s

DELTA_T = 1.0275 # seconds

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
EARTH_RADIUS = 6.3781E6  # meters


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
    while angle >= 2*math.pi:
        angle -= 2*math.pi

    while angle <= 0:
        angle += 2*math.pi
    return angle


def angle_diff(angle1, angle2):
    """
    takes difference of two angle values
    anglediff = angle1 - angle2
    """
    anglediff = angle1 - angle2

    while angle_diff >= math.pi: 
        angle_diff -= 2*math.pi

    while angle_diff <= -math.pi:
        angle_diff += 2*math.pi
    return angle_diff

def propagate_state(x_t_prev, u_t):
    """propagate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    state vector is (roll, yaw, rolldot, yawdot, v_ang, v_mag, twd, tws)^T

    u_t  (float)         -- the control input 
    control vector is (roll_meas, yaw_meas, v_ang_meas, v_mag_meas)
    """
    u_roll      = u_t[0]
    u_yaw       = u_t[1]
    x_roll      = x_t_prev[0]
    x_yaw       = x_t_prev[1]
    u_v_ang     = u_t[2]
    u_v_mag     = u_t[3]
    x_TWD       = x_t_prev[6]
    x_TWS       = x_t_prev[7]

    roll        = u_roll                                + np.random.normal(0, STDDEV_ROLL)
    yaw         = u_yaw                                 + np.random.normal(0, STDDEV_YAW)
    roll_dot    = angle_diff(roll, x_roll) / DELTA_T   + np.random.normal(0, STDDEV_ROLL_DOT)
    yaw_dot     = angle_diff(yaw, x_yaw) / DELTA_T     + np.random.normal(0, STDDEV_YAW_DOT)
    v_ang       = wrap_to_pi(u_v_ang                    + np.random.normal(0, STDDEV_V_ANG))
    v_mag       = u_v_mag                               + np.random.normal(0, STDDEV_V_MAG)
    TWD         = wrap_to_pi(x_TWD                      + np.random.normal(0, STDDEV_TWD))
    TWS         = x_TWS                                 + np.random.normal(0, STDDEV_TWS)

    x_bar_t = np.array([roll, yaw, roll_dot, yaw_dot, v_ang, v_mag, TWD, TWS])

    return x_bar_t

def prediction_step(P_t_prev, u_t):
    """Compute the prediction of PF

    Parameters:
    P_t_prev (np.array)         -- the previous state matrix, [[x, y, theta, w]^T [x, y, theta, w]^T ...]
    u_t (np.array)              -- the control input

    Returns:
    P_t_predict (np.array)      -- the predicted state matrix after propagating all particles forward in time
    """

    # Iterate through every column (particle) of the state matrix and propagate them forward.
    P_t_predict = np.zeros(shape=P_t_prev.shape)

    # every column of the particle set is a particle: a guess of 
    # (roll, yaw, rolldot, yawdat, v_ang, v_mag, twd, tws)^T

    for i in range(NUM_PARTICLES):
        P_t_predict[:-1,i] = propagate_state(P_t_prev[:-1,i], u_t) #at this point the weight stays zero

    return P_t_predict  

def test_calc_meas_prediction():
    # roll, yaw, rolldot, yawdot, v_ang, v_mag, TWD, TWS
    x_bar_t = [0, 0, 0, 0, 0, 0, 0, 1] # wind 1 m/s from the east, boat holding still, facing east
    answer = calc_meas_prediction(x_bar_t)
    print("Expected answer: [pi/2,1]")
    print(answer)

    x_bar_t = [0, 0, 0, 0, 0, 3, 0, 1] # wind 1 m/s from the east, boat moving east, facing east
    answer = calc_meas_prediction(x_bar_t)
    print("Expected answer: [pi/2,4]")
    print(answer)

    x_bar_t = [45 * np.pi/180, 0, 0, 0, 0, 3, 0, 1] # wind 1 m/s from the east, boat moving east, facing east, heeled
    answer = calc_meas_prediction(x_bar_t)
    print("Expected answer: [pi/2,4]")
    print(answer)

    x_bar_t = [45 * np.pi/180, 0, 0, 0, 0, 0, np.pi/2, 1] # wind 1 m/s from the North, boat stopped and facing east, heeled
    answer = calc_meas_prediction(x_bar_t)
    print("Expected answer: [pi,sqrt(2)/2")
    print(answer)

    x_bar_t = [45 * np.pi/180, 0, 0, 0, 0, 1, np.pi/2, 1] # wind 1 m/s from the North, boat moving and facing east, heeled
    answer = calc_meas_prediction(x_bar_t)
    print("Expected answer: [fronter left, more than 1]")
    print(answer)

    x_bar_t = [45 * np.pi/180, 0, -1, 0, 0, 0, np.pi/2, 0] # wind 0 m/s from the North, boat facing east, heeled, swinging up
    answer = calc_meas_prediction(x_bar_t)
    print("Expected answer: [pi/2, 66]")
    print(answer)

    # expect it to be (0, 1)

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
    TWD       = x_bar_t[6]
    TWS       = x_bar_t[7]

    # convert true wind from polar coordinates to cartesian
    TWS_x_comp = TWS * np.cos(TWD) # TW East
    TWS_y_comp = TWS * np.sin(TWD) # TW North

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
    z_AWA = np.arctan2(app_wind_fwd_of_vane, app_wind_right_of_vane) # note: this is in [-pi, pi]
    z_AWS = np.linalg.norm([app_wind_right_of_vane, app_wind_fwd_of_vane])

    z_bar_t = np.array([z_AWA, z_AWS])

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
    roll_mean       = np.average(P_t[0,:])
    yaw_mean        = np.average(P_t[1,:])
    roll_dot_mean   = np.average(P_t[2,:])
    yaw_dot_mean    = np.average(P_t[3,:])
    v_ang_mean      = np.average(P_t[4,:])
    v_mag_dot_mean  = np.average(P_t[5,:])
    TWD_mean        = np.average(P_t[6,:])
    TWS_mean        = np.average(P_t[7,:])

    # roll_mean       = np.average(P_t[0,:], weights=P_t[-1,:])
    # yaw_mean        = np.average(P_t[1,:], weights=P_t[-1,:])
    # roll_dot_mean   = np.average(P_t[2,:], weights=P_t[-1,:])
    # yaw_dot_mean    = np.average(P_t[3,:], weights=P_t[-1,:])
    # v_ang_mean      = np.average(P_t[4,:], weights=P_t[-1,:])
    # v_mag_dot_mean  = np.average(P_t[5,:], weights=P_t[-1,:])
    # TWD_mean        = np.average(P_t[6,:], weights=P_t[-1,:])
    # TWS_mean        = np.average(P_t[7,:], weights=P_t[-1,:])


    state_est_t = np.array([roll_mean, yaw_mean, roll_dot_mean, yaw_dot_mean, v_ang_mean, v_mag_dot_mean, TWD_mean, TWS_mean])

    return state_est_t

def correction_step(P_t_predict, z_t):
    """Compute the correction of EKF

    Parameters:
    P_t_predict         (np.array)    -- the predicted state matrix time t
    z_t                 (np.array)    -- the measured state of time t

    Returns:
    P_t                 (np.array)    -- the final state matrix estimate of time t
    """
    AWA_normal_object = scipy.stats.norm(0, STDDEV_MEAS_AWA)
    AWS_normal_object = scipy.stats.norm(0, STDDEV_MEAS_AWS)

    # Calculate weight for each particle j
    for j in range(NUM_PARTICLES):
        # calculate what the particle thinks the measurement should be 
        z_bar_t = calc_meas_prediction(P_t_predict[:-1,j])

        # find the distances between z_t and z_pred
        AWA_err = z_t[0] - z_bar_t[0]
        AWS_err = z_t[1] - z_bar_t[1]

        # weight is zero-mean normal PDF evaluated at d with stddev defined above
        AWA_weight = AWA_normal_object.pdf(AWA_err)
        AWS_weight = AWS_normal_object.pdf(AWS_err)
        comb_weight = AWA_weight * AWS_weight

        P_t_predict[-1,j] = comb_weight

    # Sample from the set of particles proportional to their weights
    # draw another card from the deck of particle-cards with probability w

    weights = P_t_predict[-1,:] / sum(P_t_predict[-1,:])
    indices = np.random.choice(range(NUM_PARTICLES), size=(NUM_PARTICLES), replace=True, p=weights)

    P_t = np.zeros(shape=P_t_predict.shape)
    for m in range(len(indices)):
        P_t[:,m] = P_t_predict[:,indices[m]]
    
    state_est_t = calc_mean_state(P_t)
    return P_t, state_est_t

def plot_graphs(time_stamps, state_estimates, z_AWA, z_AWS, data_TWD, data_TWS):
    fig, axs = plt.subplots(2,2)

    axs[0,0].plot(time_stamps, state_estimates[6,:], color='g', label='PF Estimate', linewidth=1, zorder = 10)
    axs[0,0].plot(time_stamps, data_TWD, color='r', label='Professional Estimate', linewidth=1, zorder = 0)
    axs[0,0].set(xlabel='Time (s)', ylabel='TWD (rad from east)')
    axs[0,0].set_title('TWD vs Time')
    axs[0,0].legend()


    axs[1,0].plot(time_stamps, state_estimates[7,:], color='g', label='PF Estimate', linewidth=1, zorder = 10)
    axs[1,0].plot(time_stamps, data_TWS, color='r', label='Professional Estimate', linewidth=1, zorder = 0)
    axs[1,0].set(xlabel='Time (s)', ylabel='TWS kts')
    axs[1,0].set_title('TWS vs Time')
    axs[1,0].legend()

    axs[0,1].plot(time_stamps, state_estimates[2,:], linewidth=1)
    axs[0,1].set(xlabel='Time (s)', ylabel='Roll dot (rad/s)')
    axs[0,1].set_title('Measured Roll rate vs Time')

    axs[1,1].plot(time_stamps, z_AWA, linewidth=1)
    axs[1,1].set(xlabel='Time (s)', ylabel='AWA (rad from starboard)')
    axs[1,1].set_title('Measured AWA vs Time')


    # plt.figure(1)
    # plt.plot(time_stamps, state_estimates[6,:], color='g', label='PF Estimate')
    # plt.xlabel('Time (s)')
    # plt.ylabel('TWD (rad from east)')
    # plt.title('TWD vs Time')
    # plt.legend()


    # plt.figure(2)
    # plt.plot(time_stamps, state_estimates[7,:], color='g', label='PF Estimate')
    # plt.xlabel('Time (s)')
    # plt.ylabel('TWS kts')
    # plt.title('TWS vs Time')
    # plt.legend()

    # plt.figure(3)
    # plt.plot(time_stamps, state_estimates[2,:])
    # plt.xlabel('Time (s)')
    # plt.ylabel('Roll dot (rad/s)')
    # plt.title('Roll rate vs Time')

    # plt.figure(4)
    # plt.plot(time_stamps, z_AWA)
    # plt.xlabel('Time (s)')
    # plt.ylabel('AWA (rad from starboard)')
    # plt.title('Measured AWA vs Time')

    # plt.figure(1)
    # plt.plot(time_stamps, data_TWD, color='r', label='Professional Estimate')

    # plt.figure(2)
    # plt.plot(time_stamps, data_TWS, color='r', label='Professional Estimate')


    plt.show()

def main():
    """Run a PF on sailboat data"""

    filepath = "./"
    filename = "2019Aug10_revised"
    data = load_data(filepath + filename)
    
    # remove all nan values
    #for key in data.keys():
    #    data[key] = map(lambda x: 0 if np.isnan(x) else x, data[key])

    # Load data into variables
    time_stamps = data["Timestamp"]
    lat_gps = data["Lat"]
    lon_gps = data["Lon"]
    u_roll  = [wrap_to_pi(x * np.pi/180) for x in data['Heel']]         # ref mast, to the right side of the boat
    u_yaw   = [wrap_to_pi(np.pi/2 - x*np.pi/180) for x in data['HDG']]  # HDG is changed into ref E CCW, initially imported w ref N CW 
    u_v_ang = [wrap_to_pi(np.pi/2 - x*np.pi/180) for x in data["COG"]]  # COG is changed into ref E CCW, initially imported w ref N CW
    u_v_mag = data["SOG"]          
    z_AWA = [wrap_to_pi(np.pi/2 - x*np.pi/180) for x in data["AWA"]]    # AWA is changed into ref starboard CCW, initially imported w ref forward CW
    z_AWS = data["AWS"]
    data_TWD = [wrap_to_pi(np.pi/2 - x*np.pi/180) for x in data["TWD"]] # TWD is changed into ref E CCW, initially imported w ref N CW
    data_TWS = data["TWS"]
    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]


    #  Initialize filter
    N = 8  # number of states

    # Randomly generate initial particles from a normal distribution centered around (at 0,0,0)
    # Use small STDDEV_INIT for known start position 
    # Use large STDDEV_INIT for random start position
    P_t_prev = np.random.normal(25,STDDEV_INIT, size=(N+1,NUM_PARTICLES))
    known_initial = np.array([[u_roll[0],u_yaw[0],0,0,u_v_ang[0],u_v_mag[0],data_TWD[0],data_TWS[0]]]) #initial state assum global (0,0) is at northwest corner
    for i in range(1,NUM_PARTICLES):
        P_t_prev[:-1,i] = known_initial
    P_t_prev[-1, :] = 1.0 / NUM_PARTICLES # assign equal weights to all particles


    state_estimates = np.empty((N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))
    errorsq = np.empty(len(time_stamps))
    RMSE_TWD = np.empty(len(time_stamps))
    RMSE_TWs = np.empty(len(time_stamps))

    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        print(t)
        # Get control input
        u_t = np.array([u_roll[t], u_yaw[t], u_v_ang[t], u_v_mag[t]])

        # if t == 200:
        #     pdb.set_trace()

        # Prediction Step
        P_t_predict = prediction_step(P_t_prev, u_t)

        # Get measurement
        z_t = np.array([z_AWA[t], z_AWS[t]])

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
        # errorsq_TWD[t] = (data_TWD - state_est_t[6])**2
        # errorsq_TWS[t] = (data_TWS - state_est_t[7])**2
        # RMSE_TWD[t] = np.sqrt(np.sum(errorsq_TWD[0:t]/t))
        # RMSE_TWS[t] = np.sqrt(np.sum(errorsq_TWS[0:t]/t))

        # # Plot Results
        # plt.figure(1)
        # plt.plot(gps_estimates[0],
        #                 gps_estimates[1], 'b.', label='GPS (Expected Path)',zorder=40)
        # if np.mod(t, 30) == 0:
        #     plt.figure(1)
        #     plt.quiver(state_estimates[0, t], state_estimates[1, t], np.cos(
        #         state_estimates[2, t]), np.sin(state_estimates[2, t]), color='r',label='Estimated State', zorder=50)

        #     skip_num = 40
        #     plt.scatter(P_t[0,::skip_num], P_t[1,::skip_num], color='g', label='Particles', s=2, zorder=10)
        #     plt.xlim(-4, 14)
        #     plt.ylim(-14, 4)
        #     plt.xlabel('East (m)')
        #     plt.ylabel('North (m)')
        #     if t == 0:
        #         expected_path_x = [0, 10, 10, 0, 0]
        #         expected_path_y = [0, 0, -10, -10, 0]
        #         plt.plot(expected_path_x,expected_path_y, 'k', label='Perfect Path', zorder=0, linewidth=2)
        #         plt.plot(X_LANDMARK,Y_LANDMARK, 'mo', label='Landmark Location')
        #         plt.legend()
        #     plt.pause(0.0001)

        # plt.figure(2)
        # plt.plot(t, errorsq[t], 'k.')

        # plt.xlabel('Timestep (0.1 s)')
        # plt.ylabel('Squared Error (m^2)')
        # if t == 0:
        #     plt.legend()

        # print(t)
        # plt.pause(0.0001)

    plot_graphs(time_stamps, state_estimates, z_AWA, z_AWS, data_TWD, data_TWS)
    pdb.set_trace()
    #print(RMSE[-1])

    return 0

if __name__ == "__main__":
    print = sciplot.print_wrapper(print)
    main()
