# by Paul-Ruben Schlumbom, psch250
import sys
import json
import pickle
import numpy as np
from time import time
from tsai_calibrate import get_calibration_values, apply_K1
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

def get_points(jsonpath, centre, K=None):
    """
    Load a set of calibration points from an appropriately formatted .csv file
    :param jsonpath: string, path to json file describing data
    :param centre: tuple of two floats, coordinate of image centre
    :param K: float, optional K_1 value with which to undistort pixel points in file first
    :return: numpy array of shape (N, 5) of N calibration points
    """
    with open(jsonpath, encoding='utf-8') as data:
        json_data = json.loads(data.read())

    filepath = json_data['filepath']

    points = []
    with open(filepath) as file:
        for line in file:
            points.append([float(p) for p in line.split(',')])

    points = np.asarray(points)
    if K:
        uv_K = apply_K1(points[:, [3, 4]], centre, K)
        points = np.hstack((points[:, [0, 1, 2]], uv_K))

    return points

def prune_points(ptsL, ptsR, centre):
    """
    keep only the closest 25% of points to the image centre
    :param ptsL: numpy array of shape (N,5), points in left image
    :param ptsR: numpy array of shape (N,5), points in right image
    :param centre: tuple of two floats, pixel coordinates of image centre
    :return: to numpy arrays of shape (M,5) where M<=N for the left and right image, respectively
    """
    uv_d = ptsL[:, [3, 4]]
    r_d = np.asarray([np.sqrt(np.power(p[0] - centre[0], 2) + np.power(p[1] - centre[1], 2)) for p in uv_d])

    lim = max(7, len(ptsL)//4)
    indexes = r_d.argsort()[:lim]
    newL, newR = [], []
    for ind in indexes:
        newL.append(ptsL[ind])
        newR.append(ptsR[ind])
    return np.asarray(newL), np.asarray(newR)

def get_O_I(point, vals):
    """
    Get the camera origin and image point with respect to the World Reference Frame
    :param point: array of size (2,), in the form (u, v)
    :param vals: values dictionary associated with the image point
    :return: numpy arrays, camera origin and point coordinate wrt the WRF
    """
    RTinv = np.linalg.inv(vals['RT'])
    cx, cy, dx, dy, f = vals['xc'], vals['yc'], vals['dx'], vals['dy'], vals['f']

    O_c_w = np.matmul(RTinv, np.array([0, 0, 0, 1]))

    I_i_c = (np.array([point[0], point[1], 1, 1]) - np.array([cx, cy, 0, 0])) * np.array([dx, dy, f, 1])

    I_i_w = np.matmul(RTinv, I_i_c)

    return O_c_w, I_i_w

def get_intersection(Lpt, Rpt, Lvals, Rvals):
    """
    Get the intersection of two rays projected through the given points of the given image planes
    :param Lpt: Array of shape (5,) in the form (X,Y,Z,u,v) as read from the .CSV files, for the left image point
    :param Rpt: Array of shape (5,) in the form (X,Y,Z,u,v) as read from the .CSV files, for the right image point
    :param Lvals: values dictionary of the left camera
    :param Rvals: values dictionary of the right camera
    :return: array of shape (3,) in the form (X,Y,Z) measured in mm
    """
    p1, p2 = get_O_I(Lpt[3:5], Lvals)
    p3, p4 = get_O_I(Rpt[3:5], Rvals)

    D21 = p2 - p1
    D43 = p4 - p3
    D13 = p1 - p3

    M = np.array([[D21.dot(D21), -D21.dot(D43)],
                  [-D21.dot(D43), D43.dot(D43)]])
    N = np.array([[D13.dot(D21)],
                  [-D13.dot(D43)]])

    Ts = np.matmul(-np.linalg.inv(M), N)
    a = p1 + (Ts[0] * D21)
    b = p3 + (Ts[1] * D43)
    ret = (a + b) / 2
    ret /= ret[-1]
    return ret[:3]

def get_predicted_points(Lpts, Rpts, Lvals, Rvals):
    """
    Given set of left & right calibration point datasets and associated camera parameters, reconstruct the original 3D points
    :param Lpts: numpy array of shape (N,5), points in left image
    :param Rpts: numpy array of shape (N,5), points in right image
    :param Lvals: values dictionary for left camera
    :param Rvals: values dictionary for right camera
    :return: numpy array of shape (N,3), the reconstructed 3D coordinates
    """
    predicted_points = []
    for i in range(len(Lpts)):
        predicted_points.append(get_intersection(Lpts[i], Rpts[i], Lvals, Rvals))
    predicted_points = np.asarray(predicted_points)
    return predicted_points

def get_stereo_error(Lpts, Rpts, Lvals, Rvals):
    """
    Calculate the stereo error of the camera system
    :param Lpts: Array of shape (N, 5) containing the calibration points as measured for the left image in the format (X,Y,Z,u,v)
    :param Rpts: Array of shape (N, 5) containing the calibration points as measured for the right image in the format (X,Y,Z,u,v)
    :param Lvals: dictionary of values for the left camera system
    :param Rvals: dictionary of values for the right camera system
    :return: mean stereo error of the system
    """
    predicted_points = []
    for i in range(len(Lpts)):
        predicted_points.append(get_intersection(Lpts[i], Rpts[i], Lvals, Rvals))
    predicted_points = np.asarray(predicted_points)
    errors = (predicted_points - Lpts[:, [0, 1, 2]])**2
    errors = [np.sqrt(np.sum(p)) for p in errors]
    error = np.mean(errors)
    return error

def plot_3D(points):
    """
    plot 3D points in 3D
    :param points: numpy array of shape (N,3+)
    :return: NA
    """
    xs = np.reshape(points[:, [0]], (len(points)))
    ys = np.reshape(points[:, [1]], (len(points)))
    zs = np.reshape(points[:, [2]], (len(points)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)
    plt.show()

def plot_double_3D(a, b):
    """
    Plot 2 sets of points in 3D space
    :param a: numpy array of shape (N,3+)
    :param b: numpy array of shape (N,3+)
    :return: NA
    """
    axs = np.reshape(a[:, [0]], (len(a)))
    ays = np.reshape(a[:, [1]], (len(a)))
    azs = np.reshape(a[:, [2]], (len(a)))

    bxs = np.reshape(b[:, [0]], (len(b)))
    bys = np.reshape(b[:, [1]], (len(b)))
    bzs = np.reshape(b[:, [2]], (len(b)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(axs, ays, azs, c='b')
    ax.scatter(bxs, bys, bzs, c='r')
    plt.show()

def least_stereo_ftz(ftzs, Lpts, Rpts, Lvals, Rvals):
    """
    calculate stereo error for new values of f, t_z
    :param ftzs: numpy array of shape (2,) in the form: [f, t_z]
    :param Lpts: numpy array of shape (N,5), points in left image
    :param Rpts: numpy array of shape (N,5), points in right image
    :param Lvals: values dictionary for left camera
    :param Rvals: values dictionary for right camera
    :return: float, mean stereo error
    """
    Lv = Lvals.copy()
    Rv = Rvals.copy()
    Lv['f'] = ftzs[0]
    Rv['f'] = ftzs[1]
    Lv['tz'] = ftzs[2]
    Rv['tz'] = ftzs[3]
    return get_stereo_error(Lpts, Rpts, Lv, Rv)

def least_stereo_Kftz(Kftzs, Lpts, Rpts, Lvals, Rvals):
    """
    calculate stereo error for new values of K_1, f, t_z
    :param Kftzs: numpy array of shape (3,) in the form: [K_1, f, t_z]
    :param Lpts: numpy array of shape (N,5), points in left image
    :param Rpts: numpy array of shape (N,5), points in right image
    :param Lvals: values dictionary for left camera
    :param Rvals: values dictionary for right camera
    :return: float, mean stereo error
    """
    Lcent = (Lvals['xc'], Lvals['yc'])
    Rcent = (Rvals['xc'], Rvals['yc'])
    LK = Kftzs[0]
    RK = Kftzs[1]
    Luv_K = apply_K1(Lpts[:, [3, 4]], Lcent, LK)
    LKpts = np.hstack((Lpts[:, [0, 1, 2]], Luv_K))
    Ruv_K = apply_K1(Rpts[:, [3, 4]], Rcent, RK)
    RKpts = np.hstack((Rpts[:, [0, 1, 2]], Ruv_K))


    Lv = Lvals.copy()
    Rv = Rvals.copy()
    Lv['f'] = Kftzs[2]
    Rv['f'] = Kftzs[3]
    Lv['tz'] = Kftzs[4]
    Rv['tz'] = Kftzs[5]
    return get_stereo_error(LKpts, RKpts, Lv, Rv)

def save_model(jsonpath, vals):
    """
    save the values dictionary and relevant transformation matrixes to .pickle file
    :param jsonpath: path to json file containing all information about the experiment
    :param vals: values dictionary
    :return: NA
    """
    with open(jsonpath, encoding='utf-8') as data:
        json_data = json.loads(data.read())

    model_name = json_data['model_name']

    out_dict = {'full_transformation_matrix': vals['full_transform'], 'RT': vals['RT'],
                'values_dict': vals}
    f = open('models/' + model_name + '.pickle', 'wb')
    pickle.dump(out_dict, f)
    f.close()

if __name__ == "__main__":
    optimise = False
    undistort = True
    plot_result = False
    Lpath, Rpath = sys.argv[1], sys.argv[2]
    #Lpath = 'input/set3-L.json'
    #Rpath = 'input/set3-R.json'
    L_vals = get_calibration_values(Lpath, report=False, optimise=optimise, undistort=undistort).copy()
    R_vals = get_calibration_values(Rpath, report=False, optimise=optimise, undistort=undistort).copy()
    print('---')
    print("L side: pixel error {:.2f}px, cube error {:.2f}mm.".format(L_vals['pixel_error'], L_vals['cube_error']))
    print("R side: pixel error {:.2f}px, cube error {:.2f}mm.".format(R_vals['pixel_error'], R_vals['cube_error']))
    print("---")

    # load the points data
    if undistort:
        L_points = get_points(Lpath, (L_vals['xc'], L_vals['yc']), L_vals['K1'])
        R_points = get_points(Rpath, (R_vals['xc'], R_vals['yc']), R_vals['K1'])
    else:
        L_points = get_points(Lpath, (L_vals['xc'], L_vals['yc']))
        R_points = get_points(Rpath, (R_vals['xc'], R_vals['yc']))
    L_points, R_points = prune_points(L_points, R_points, (L_vals['xc'], L_vals['yc']))

    # baseline calculation
    baseline = np.sqrt(np.sum((L_vals['T'] - R_vals['T'])**2))
    print("original baseline: ", baseline, " mm")

    # stereo error calculation
    err = get_stereo_error(L_points, R_points, L_vals, R_vals)
    print("original stereo error: ", err, " mm")
    print("---")

    # plotting
    newLpts = get_points(Lpath, (L_vals['xc'], L_vals['yc']), L_vals['K1'])
    newRpts = get_points(Rpath, (R_vals['xc'], R_vals['yc']), R_vals['K1'])
    if optimise: newLpts, newRpts = prune_points(newLpts, newRpts, (L_vals['xc'], L_vals['yc']))

    pred_pts = get_predicted_points(L_points, R_points, L_vals, R_vals)

    if plot_result: plot_double_3D(L_points, pred_pts)

    # optimise results
    ftzs = np.array([L_vals['f'], R_vals['f'], L_vals['tz'], R_vals['tz']])
    result = least_squares(least_stereo_ftz, ftzs, args=(L_points, R_points, L_vals, R_vals))
    L_vals['f'] = result.x[0]
    R_vals['f'] = result.x[1]
    L_vals['tz'] = result.x[2]
    R_vals['tz'] = result.x[3]
    err = get_stereo_error(L_points, R_points, L_vals, R_vals)
    baseline = np.sqrt(np.sum((L_vals['T'] - R_vals['T']) ** 2))
    print("optimised baseline: ", baseline, " mm")
    print("optimised stereo error: ", err, " mm")


    #plotting

    pred_pts = get_predicted_points(L_points, R_points, L_vals, R_vals)

    if plot_result: plot_double_3D(L_points, pred_pts)

    save_model(Lpath, L_vals)
    save_model(Rpath, R_vals)