# by Paul-Ruben Schlumbom, psch250
import sys
import json
import pickle
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

report = False	# whether or not to print extra info
d_x, d_y, c_x0, c_y0, f = 1, 1, 1, 1, 1
values_dict = {}

def std_error(vals):
    return np.std(vals)/np.sqrt(len(vals))

def let_there_be_M(xd, yd, xyzw):
    M = [np.concatenate((yd[i]*xyzw[i], [yd[i]], -xd[i]*xyzw[i])) for i in range(len(xd))]
    return np.asarray(M)

def error(e, m):
    return np.sqrt(np.sum((e - m)**2))

def print_dict(in_dict):
    for key in sorted(in_dict.keys()):
        print("{} = {}".format(key, in_dict[key]))

def apply_K1(points, centre, K1):
    """
    compensate for first-order radial distortion
    :param points: numpy array of shape (N,2) of calibration image points
    :param centre: tuple of 2 floats, pixel coordinates of image centre
    :param K1: float, K_1 value to use
    :return: numpy array of shape (N,2) of undistorted pixel coordinates
    """
    cpoints = np.asarray([[p[0] - centre[0], p[1] - centre[1]] for p in points])
    r_d = np.asarray([np.sqrt(np.power(p[0] - centre[0], 2) + np.power(p[1] - centre[1], 2)) for p in points])
    r_d = r_d.reshape((len(r_d), 1))
    modpts = cpoints * (1 + K1 * np.power(r_d, 2))
    retpts = np.asarray([[p[0] + centre[0], p[1] + centre[1]] for p in modpts])
    return retpts

def project_point(point, RTinv, values_dict):
    """
    project a ray from the camera origin through the given point in the image plane into the cube plane
    :param point: numpy array of shape (5,) of some given calibration data point in format (X,Y,Z,u,v)
    :param RTinv: inverse rotation-transformation matrix of camera
    :param values_dict: dictionary of calibration parameter values
    :return: numpy array of shape (3,) of point back projected into cube plane
    """
    d_x, d_y, c_x0, c_y0, f = values_dict['dx'], values_dict['dy'], values_dict['xc'], values_dict['yc'], values_dict['f']

    t_cube_pt = point[:3]
    t_im_pt = point[3:]

    O_c_c = np.asarray([0, 0, 0, 1])
    O_c_w = np.matmul(RTinv, O_c_c)
    #print(O_c_w)

    # 2. get I_i^c from pixel point
    #print(c_x0, c_y0, d_x, d_y, f)
    #print(type(t_im_pt[0]), type(c_x0), type(d_x))
    ix = (t_im_pt[0] - c_x0) * d_x
    iy = (t_im_pt[1] - c_y0) * d_y
    #print(type(ix), type(iy), type(f), type(1))
    I_i_c = np.asarray([ix, iy, f, 1])
    #print(I_i_c) # here things go wrong

    # 3. calculate I_i^w, i.e. pixel point in WRF
    #print(type(RTinv), type(I_i_c))
    I_i_w = np.matmul(RTinv, I_i_c)
    #print(I_i_w)

    # 4. determine t for parametric line equation
    # first, find plane of intersection; i.e. which index = 0
    for i in range(len(t_cube_pt)):
        if t_cube_pt[i] == 0:
            plane_axis = i
    #print(t_cube_pt, plane_axis)
    t = -O_c_w[plane_axis] / (I_i_w[plane_axis] - O_c_w[plane_axis])
    #print(t)
    e_pt = []
    for i in range(len(t_cube_pt)):
        if i == plane_axis:
            e_pt.append(0)
        else:
            component = O_c_w[i] + t*(I_i_w[i] - O_c_w[i])
            e_pt.append(component)
    e_pt = np.asarray(e_pt)
    #print(e_pt, t_cube_pt, e_pt - t_cube_pt)
    #print('---')
    #if report:
        #print(e_pt, t_cube_pt, e_pt - t_cube_pt)
    return e_pt

def get_setup_values(input_path):
    """
    get setup values from given json file
    :param input_path: string, path to json file
    :return: tuple, relevant values
    """
    # SETUP
    get_manual = False
    known_f = False
    f = 1

    try:
        jsonFile = input_path
        with open(jsonFile, encoding='utf-8') as data:
            json_data = json.loads(data.read())
    except:
        if 'y' in input("WARNING COULD NOT LOAD JSON. ENTER MANUAL VALUES INSTEAD? y/n: "):
            get_manual = True
        else:
            print("STOPPING")
            sys.exit()

    if get_manual:
        c_width = int(input("Enter image x-dimension (width) in pixels: "))
        c_height = int(input("Enter image y-coordinate (height) in pixels: "))
        if 'y' in input("Focal length known? y/n: "):
            f = float(input("Enter focal length in mm: "))
            known_f = True
        d_x = float(input("Enter d_x in mm: "))
        d_y = float(input("Enter d_y in mm: "))
        filepath = input("Enter path of CSV file: ")
        model_name = input("Name of model to save: ")
    else:
        c_width = json_data['i_width']
        c_height = json_data['i_height']
        if json_data['f'] != -1:
            f = json_data['f']
            known_f = True
        d_x = json_data['d_x']
        d_y = json_data['d_y']
        filepath = json_data['filepath']
        model_name = json_data['model_name']

    # note: Hiroki's camera has image dimensions 4224x3200. The image centre is at 2112, 1600
    c_x0 = int(c_width / 2)
    c_y0 = int(c_height / 2)
    return filepath, d_x, d_y, c_x0, c_y0, f, known_f, model_name

def get_calibration_matrix(points, d_x, d_y, c_x0, c_y0, f, known_f):
    """
    Use Tsai's calibration technique to determine a camera calibration matrix
    :param points:	list of points extracted from CSV file, where each point is (X, Y, Z, u, v)
    :param d_x: distance between pixels in x-direction (mm)
    :param d_y: distance between pixels in y-direction (mm)
    :param c_x0: x-centre of image in pixels
    :param c_y0: y-centre of image in pixels
    :param f: focal length (=1 if unknown)
    :param known_f: boolean; whether or not f is already known
    :return: completed calibration matrix
    """

    # prepare point matrices
    XYZ_w = np.asarray([p[:3] for p in points])
    c_x = np.asarray([p[3] for p in points])
    c_y = np.asarray([p[4] for p in points])

    # STEP 1: coordinates in distorted image
    s_x = 1
    s_y = 1
    x_d = d_x * (c_x - c_x0) / s_x
    y_d = d_y * (c_y - c_y0) / s_y

    # STEP 2: finding the 7 parameters of L
    # create M (nx7 matrix) to calculate L (7 values)
    # M = [m_1, m_2, m_3, ..., m_n]^T
    # m_i = [y_d_i*X_w_i,	 y_d_i*Y_w_i,	 y_d_i*Z_w_i,	 y_d_i,	   -x_d_i*X_w_i,	-x_d_i*Y_w_i,	 -x_d_i*Z_w_i]
    M = let_there_be_M(x_d, y_d, XYZ_w)

    # get inverse of M
    MT = np.transpose(M)
    Minv = np.matmul(MT, M)
    Minv = np.linalg.inv(Minv)
    Minv = np.matmul(Minv, MT)

    # get L
    L = np.matmul(Minv, x_d)

    # STEP 3: find	t_y	 and scaling factor	 s_x
    a_1 = L[0]
    a_2 = L[1]
    a_3 = L[2]
    a_4 = L[3]
    a_5 = L[4]
    a_6 = L[5]
    a_7 = L[6]
    t_y_abs = 1 / np.sqrt((a_5 ** 2) + (a_6 ** 2) + (a_7 ** 2))

    s_x = t_y_abs * np.sqrt((a_1 ** 2) + (a_2 ** 2) + (a_3 ** 2))

    # STEP 4: find the sign of	t_y
    # find reference 3D point most distant from image centre
    most_distant_point = max(points, key=lambda x: np.sqrt(((x[3] - c_x0) ** 2) + ((x[4] - c_y0) ** 2)))

    # compute the parameters
    r_11 = a_1 * t_y_abs
    r_12 = a_2 * t_y_abs
    r_13 = a_3 * t_y_abs
    r_21 = a_5 * t_y_abs
    r_22 = a_6 * t_y_abs
    r_23 = a_7 * t_y_abs
    t_x = a_4 * t_y_abs

    # determine whether or not t_y should in fact be negative
    temp_var = np.concatenate((most_distant_point[:2], [most_distant_point[1]], [1]))
    compare_x = (np.sign(np.dot(temp_var, [r_11, r_12, r_13, t_x])) == np.sign(most_distant_point[3]))
    compare_y = (np.sign(np.dot(temp_var, [r_21, r_22, r_23, t_y_abs])) == np.sign(most_distant_point[4]))
    if not (compare_x and compare_y):
        if report: print("swapping sign!")
        t_y = t_y_abs * -1
    else:
        if report: print("no sign swap needed")
        t_y = t_y_abs

    # STEP 5: calculate remaining 3 components of rotation matrix
    # recalculate remaining components of rotation matrix R and translation vector t:
    r_11 = a_1 * t_y / s_x
    r_12 = a_2 * t_y / s_x
    r_13 = a_3 * t_y / s_x
    r_21 = a_5 * t_y
    r_22 = a_6 * t_y
    r_23 = a_7 * t_y
    t_x = a_4 * t_y / s_x

    # calculate remaining 3 components
    #  need to calculate lambda first from determinants; we have lambda = 1 / sqrt(d1^2 + d2^2 + d3^2)
    # determinant of 2x2 matrix [[a, b], [c, d]] = ad - bc
    d_1 = (r_12 * r_23) - (r_13 * r_22)
    d_2 = (r_13 * r_21) - (r_11 * r_23)
    d_3 = (r_11 * r_22) - (r_12 * r_21)
    lambda_var = 1 / np.sqrt((d_1 ** 2) + (d_2 ** 2) + (d_3 ** 2))

    r_31 = lambda_var * d_1
    r_32 = lambda_var * d_2
    r_33 = lambda_var * d_3

    # STEP 6: approximate focal length	f  and Z-coordinate translation factor	t_z
    U_z = np.matmul(XYZ_w, np.transpose([r_31, r_32, r_33]))

    extra_ones = np.ones((len(U_z), 1))
    xyz_extra_ones = np.append(XYZ_w, extra_ones, axis=1)
    U_y = np.matmul(xyz_extra_ones, np.transpose([r_21, r_22, r_23, t_y]))

    MT = np.asarray([U_y, -1 * y_d])

    mT = U_z * y_d	# simple side-by-side multiplication, creating Nx1 matrix [U_z1*y_d1, U_z2*y_d2, ..., U_zn*y_dn]

    f_t_soln = np.matmul(np.matmul(np.linalg.inv(np.matmul(MT, np.transpose(MT))), MT), mT)
    if not known_f:
        f = f_t_soln[0]
    t_z = f_t_soln[1]

    # Finally, get whole transformation matrix
    K_1 = np.asarray([[1 / d_x, 0, c_x0],
                      [0, 1 / d_y, c_y0],
                      [0, 0, 1]])
    K_2 = np.asarray([[f, 0, 0, 0],
                      [0, f, 0, 0],
                      [0, 0, 1, 0]])
    RT = np.asarray([[r_11, r_12, r_13, t_x],
                     [r_21, r_22, r_23, t_y],
                     [r_31, r_32, r_33, t_z],
                     [0, 0, 0, 1]])

    full_transform_matrix = np.matmul(K_1, np.matmul(K_2, RT))
    full_transform_matrix /= full_transform_matrix[2][3]

    values_dict['r11'] = r_11
    values_dict['r12'] = r_12
    values_dict['r13'] = r_13
    values_dict['r21'] = r_21
    values_dict['r22'] = r_22
    values_dict['r23'] = r_23
    values_dict['r31'] = r_31
    values_dict['r32'] = r_32
    values_dict['r33'] = r_33
    values_dict['tx'] = t_x
    values_dict['ty'] = t_y
    values_dict['tz'] = t_z
    values_dict['f'] = f
    values_dict['dx'] = d_x
    values_dict['dy'] = d_y
    values_dict['xc'] = c_x0
    values_dict['yc'] = c_y0
    values_dict['a_1'] = a_1
    values_dict['a_2'] = a_2
    values_dict['a_3'] = a_3
    values_dict['a_4'] = a_4
    values_dict['a_5'] = a_5
    values_dict['a_6'] = a_6
    values_dict['a_7'] = a_7

    T = np.asarray([[t_x],
                    [t_y],
                    [t_z],
                    [1]])
    R = np.asarray([[r_11, r_12, r_13, 0],
                    [r_21, r_22, r_23, 0],
                    [r_31, r_32, r_33, 0],
                    [0, 0, 0, 1]])

    values_dict['T'] = T
    values_dict['R'] = R

    values_dict['full_transform'] = full_transform_matrix
    values_dict['RT'] = RT

    return values_dict

def estimate_k1(points, values_dict):
    """
    produce a first estimate of K_1 using the points data and calibration parameters
    :param points: numpy array of shape (N,5) of calibration points
    :param values_dict: dictionary of calibration parameter values
    :return: tuple: original pixel error, original cube error, estimate of K_1, new cube error, new cube individual errors
    """
    cx, cy, dx, dy = values_dict['xc'], values_dict['yc'], values_dict['dx'], values_dict['dy']
    full_matrix = values_dict['full_transform']
    cube_error, _, _, _ = get_cube_error(points, values_dict, silenced=True)

    # 1. calculate r_d for every point
    uv_d = np.asarray([[p[3], p[4]] for p in points])
    r_d = np.asarray([np.sqrt(np.power(p[0] - cx, 2) + np.power(p[1] - cy, 2)) for p in uv_d])

    # 2. calculate r_u for every point
    # Begin by getting the pixel errors
    pixel_error, pixel_errors, uv_u, pixel_errors_per_dimension = get_pixel_error(full_matrix, points, silenced=True)

    r_u = np.asarray([np.sqrt(np.power(p[0] - cx, 2) + np.power(p[1] - cy, 2)) for p in uv_u])

    # 3. estimate K_1 for each point
    K1_e = (r_u - r_d) / np.power(r_d, 3)

    # 4. pick 10 most distant points (if possible) and calculate their mean K_1
    K1_mean = np.mean(np.asarray([x for _, x in sorted(zip(r_d, K1_e), reverse=True)][:min(10, len(r_d))]))

    # 5. calculate the undistorted versions of u and v using K_1
    r_d = r_d.reshape((len(r_d), 1))
    uv_d = np.asarray(uv_d)
    uv_K = uv_d * (1 + K1_mean * np.power(r_d, 2))

    # 6. calculate the cube errors of the points estimated by K_1
    # Creating new dataset of poits in format accepted by cube error function, substituting u,v values estimated by K1 into the original u,v values
    new_k1_points = np.asarray([[points[i][0], points[i][1], points[i][2], uv_K[i][0], uv_K[i][1]] for i in range(len(r_d))])

    K1_cube_error, K1_cube_errors, cb_e, cube_errors_per_dimension = get_cube_error(new_k1_points, values_dict, silenced=True)

    return pixel_error, cube_error, K1_mean, K1_cube_error, K1_cube_errors

def get_pixel_error(full_transform_matrix, points, silenced=False, K1=None, r_d=None):
    # calculate pixel errors
    xyz_extra_ones = np.hstack((points[:, [0, 1, 2]], np.ones((len(points), 1))))

    if K1:
        uv_K = points[:, [3, 4]] * (1 + K1 * np.power(r_d, 2))
        new_pts = np.hstack((points[:, [0, 1, 2]], uv_K))
    else:
        new_pts = points

    errors = []
    x_diff = []
    y_diff = []
    estimated_points = []
    for i, point in enumerate(xyz_extra_ones):
        result = np.matmul(full_transform_matrix, point)
        result /= result[-1]
        correct = new_pts[i][3:]
        errors.append(error(result[:2], correct))
        x_diff.append(np.abs(result[0] - correct[0]))
        y_diff.append(np.abs(result[1] - correct[1]))
        estimated_points.append(result[:2])
        if report:
            print(x_diff[-1], y_diff[-1])
    errors = np.asarray(errors)
    pixel_error = np.mean(errors)
    if not silenced:
        print("mean pixel error: {}\nmean x error: {}\nmean y error: {}".format(pixel_error, np.mean(x_diff), np.mean(y_diff)))
    return pixel_error, errors, estimated_points, (x_diff, y_diff)

def new_pixel_error(ptsA, ptsB):
    """
    calculate mean euclidean distance between two arrays of equivalent shape
    :param ptsA: numpy array of shape (a,b,...)
    :param ptsB: numpy array of shape (a,b,...)
    :return: mean error, individual errors, estimated points (ptsB), individual errors in each dimension
    """
    pixel_error = compute_pt_diff(ptsA, ptsB)
    pixel_errors = np.asarray([error(ptsA[i], ptsB[i]) for i in range(len(ptsA))])
    estimated_pixels = ptsB
    pixel_errors_per_dimension = (ptsA[:, [0]]-ptsB[:, [0]], ptsA[:, [1]]-ptsB[:, [1]])
    return pixel_error, pixel_errors, estimated_pixels, pixel_errors_per_dimension

def get_cube_error(points, values_dict, silenced=False):
    """
    calculate the cube backprojection error for some given set of points and calibration parameters
    :param points: numpy array of shape (N,5) of calibration points
    :param values_dict: dictionary of calibration parameter values
    :param silenced: bool, whether or not to print information about calculation process and results
    :return: mean cube error, individual cube errors, estimated 3D cube points, tuple of individual errors in each dimension
    """
    RT = values_dict['RT']
    RTinv = np.linalg.inv(RT)
    errors = []
    x_diff = []
    y_diff = []
    z_diff = []
    estimated_points = []

    for point in points:
        #print(type(point))
        estimated_point = project_point(point, RTinv, values_dict)
        errors.append(error(estimated_point, point[:3]))
        diff = np.abs(estimated_point - point[:3])
        x_diff.append(diff[0])
        y_diff.append(diff[1])
        z_diff.append(diff[2])
        estimated_points.append(estimated_point)
        if report:
            #print(diff)
            print(estimated_point)
    errors = np.asarray(errors)
    cube_error = np.mean(errors)
    if not silenced:
        print("mean cube error: {}\nmean x error: {}\nmean y error: {}\nmean z error: {}".format(cube_error, np.mean(x_diff),
                                                                                np.mean(y_diff), np.mean(z_diff)))
    return cube_error, errors, estimated_points, (x_diff, y_diff, z_diff)

def compute_undistorted(values_dict, points):
    """
    calculate new undistorted points using new calibration parameter values
    :param values_dict: dictionary of calibration parameter values
    :param points: numpy array of shape (N,5) of calibration points
    :return: numpy array of shape (N,2) of undistorted image coordinates
    """
    A = np.asarray([[1 / values_dict['dx'], 0, values_dict['xc']],
                    [0, 1 / values_dict['dy'], values_dict['yc']],
                    [0, 0, 1]])
    B = np.asarray([[values_dict['f'], 0, 0, 0],
                    [0, values_dict['f'], 0, 0],
                    [0, 0, 1, 0]])

    new_full = np.matmul(A, np.matmul(B, values_dict['RT']))
    if new_full[2][3] == 0:
        new_full[2][3] = 1e-10
    new_full /= new_full[2][3]

    xyz_extra_ones = np.hstack((points[:, [0, 1, 2]], np.ones((len(points), 1))))
    estimated = [np.matmul(new_full, pt) for pt in xyz_extra_ones]
    estimated = [pt / pt[-1] for pt in estimated]
    return np.asarray([pt[:2] for pt in estimated])

def compute_pt_diff(ptsA, ptsB):
    errors = np.asarray([error(ptsA[i], ptsB[i]) for i in range(len(ptsA))])
    return np.mean(errors)

def tsai_test_radius(partition, values_dict, points, K=None):
    new_cube_error, _, _, _ = get_cube_error(points[:int(partition)], values_dict, silenced=True)
    return new_cube_error

def tsai_test_K(K, values_dict, points, t=None):
    uv_K = apply_K1(points[:, [3, 4]], (values_dict['xc'], values_dict['yc']), K)

    new_k_pts = np.hstack((points[:, [0, 1, 2]], uv_K))

    #new_cube_error, _, _, _ = get_cube_error(new_k_pts, values_dict, silenced=True)

    test_error = compute_pt_diff(new_k_pts[:, [3, 4]], compute_undistorted(values_dict, points))
    #return new_cube_error
    return test_error

def tsai_test_f(f, values_dict, points, K=None):
    if K:
        uv_K = apply_K1(points[:, [3, 4]], (values_dict['xc'], values_dict['yc']), K)
        kpoints = np.hstack((points[:, [0, 1, 2]], uv_K))
    else:
        kpoints = points

    new_vals = values_dict.copy()
    new_vals['f'] = f

    #for pt in points:
    #    print(type(pt[0]))

    #new_cube_error, _, _, _ = get_cube_error(points, new_vals, silenced=True)
    test_error = compute_pt_diff(kpoints[:, [3, 4]], compute_undistorted(new_vals, points))
    #return new_cube_error
    return test_error

def tsai_test_tz(tz, values_dict, points, K=None):
    if K:
        uv_K = apply_K1(points[:, [3, 4]], (values_dict['xc'], values_dict['yc']), K)
        kpoints = np.hstack((points[:, [0, 1, 2]], uv_K))
    else:
        kpoints = points

    new_vals = values_dict.copy()
    new_vals['tz'] = tz

    newRT = values_dict['RT'].copy()
    newRT[2][3] = tz  # tz

    new_vals['RT'] = newRT

    #new_cube_error, _, _, _ = get_cube_error(points, new_vals, silenced=True)
    test_error = compute_pt_diff(kpoints[:, [3, 4]], compute_undistorted(new_vals, points))
    #return new_cube_error
    return test_error

def tsai_test_all(theta, values_dict, points):
    K = theta[0]
    f = theta[1]
    tz = theta[2]
    newVals = values_dict.copy()

    uv_K = apply_K1(points[:, [3, 4]], (values_dict['xc'], values_dict['yc']), K)
    kpoints = np.hstack((points[:, [0, 1, 2]], uv_K))

    newVals['f'] = f

    newVals['tz'] = tz

    newRT = values_dict['RT'].copy()
    newRT[2][3] = tz  # tz

    newVals['RT'] = newRT

    test_error = compute_pt_diff(kpoints[:, [3, 4]], compute_undistorted(newVals, points))
    return test_error

def tsai_test_ftz(theta, values_dict, points, K):
    f = theta[0]
    tz = theta[1]
    newVals = values_dict.copy()

    uv_K = apply_K1(points[:, [3, 4]], (values_dict['xc'], values_dict['yc']), K)
    kpoints = np.hstack((points[:, [0, 1, 2]], uv_K))

    newVals['f'] = f

    newVals['tz'] = tz

    newRT = values_dict['RT'].copy()
    newRT[2][3] = tz  # tz

    newVals['RT'] = newRT

    test_error = compute_pt_diff(kpoints[:, [3, 4]], compute_undistorted(newVals, points))
    return test_error

def iterate(points, values_dict, vals, test_fun, i=0, init_bound=None):
    val = vals[i]
    verbose = False
    if not init_bound: bounds = (-2 * val, 0, 2 * val)
    else: bounds = (init_bound[0], (init_bound[0] + init_bound[1]) / 2, init_bound[1])
    results = (test_fun(bounds[0], values_dict, points, vals[0]), test_fun(bounds[1], values_dict, points, vals[0]), test_fun(bounds[2], values_dict, points, vals[0]))

    if results[0] > results[2]:
        bounds = (bounds[1], (bounds[1] + bounds[2]) / 2, bounds[2])
    else:
        bounds = (bounds[0], (bounds[0] + bounds[1]) / 2, bounds[1])

    count = 0
    while abs(abs(bounds[2]) - abs(bounds[0])) > abs(val / 1e8) and count < 100:
        count += 1
        if verbose: print("iteration ", count)  # , "\ndifference: ", abs(abs(bounds[2]) - abs(bounds[0])))
        results = (test_fun(bounds[0], values_dict, points, vals[0]), test_fun(bounds[1], values_dict, points, vals[0]), test_fun(bounds[2], values_dict, points, vals[0]))
        if results[0] > results[2]:
            bounds = (bounds[1], (bounds[1] + bounds[2]) / 2, bounds[2])
        else:
            bounds = (bounds[0], (bounds[0] + bounds[1]) / 2, bounds[1])
    return np.mean(results), bounds[0]

def get_optimised_values(points, values_dict, report=False):
    """
    carry out optimisation
    :param points: numpy array of shape (N,5) of calibration points
    :param values_dict: dictionary of calibration parameter values
    :param report: bool, whether or not to print information on procedure and intermediate results
    :return: new dictionary of optimised calibration parameter values
    """
    cx, cy = values_dict['xc'], values_dict['yc']
    """uv_d = points[:, [3, 4]]
    r_d = np.asarray([np.sqrt(np.power(p[0] - cx, 2) + np.power(p[1] - cy, 2)) for p in uv_d])
    r_d = r_d.reshape((len(r_d), 1))

    ordered_points = np.asarray([y for x, y in sorted(zip(r_d, points))]) # sort measured points in order of ascending radial extension
    r_d.sort()"""
    """ordered_points = points

    theta = np.array([values_dict['K1'], values_dict['f'], values_dict['tz']])
    #bestSetError, newPartition = iterate(ordered_points, values_dict, theta, tsai_test_radius, init_bound=(int(len(points) / 4), len(points)))
    newPartition = max(len(points)//4, 7)#int(newPartition)"""
    """bestKerror, newK = iterate((ordered_points[:newPartition], r_d[:newPartition]), values_dict, theta, tsai_test_K, i=0)
    bestferror, newf = iterate(ordered_points[:newPartition], values_dict, theta, tsai_test_f, i=1)
    besttzerror, newtz = iterate(ordered_points[:newPartition], values_dict, theta, tsai_test_tz, i=2)
    newTheta = np.array([newK, newf, newtz])
    bestferrorNew, newnewf = iterate(ordered_points[:newPartition], values_dict, newTheta, tsai_test_f, i=1)"""
    """bestKerror, newK = iterate(ordered_points, values_dict, theta, tsai_test_K,i=0)
    bestferror, newf = iterate(ordered_points, values_dict, theta, tsai_test_f, i=1)
    besttzerror, newtz = iterate(ordered_points, values_dict, theta, tsai_test_tz, i=2)
    newTheta = np.array([newK, newf, newtz])
    bestferrorNew, newnewf = iterate(ordered_points, values_dict, newTheta, tsai_test_f, i=1)
    best = min(bestKerror, bestferror, besttzerror, bestferrorNew)
    if report:  print("best K: {} for K={}\nbest f: {} for f={}\nbest tz:{} for tz={}\nsecond best f: {} for f={}\nbest overall: {}\npartition: {}".format(bestKerror, newK, bestferror, newf, besttzerror, newtz, bestferrorNew, newnewf, best, newPartition))

    if best == bestKerror:
        values_dict['K1'] = newK
    elif best == bestferror:
        values_dict['f'] = newf
    elif best == besttzerror:
        values_dict['tz'] = newtz
    else:
        values_dict['K1'] = newK
        values_dict['f'] = newnewf
        values_dict['tz'] = newtz"""

    theta = np.array([values_dict['K1'], values_dict['f'], values_dict['tz']])

    #ftzs = np.array([L_vals['f'], R_vals['f'], L_vals['tz'], R_vals['tz']])
    result = least_squares(tsai_test_ftz, theta[1:], args=(values_dict, points, theta[0]))
    if report:
        print(result.x)
        print(result.fun)
    theta[1] = result.x[0]
    theta[2] = result.x[1]

    values_dict['f'] = result.x[0]
    values_dict['tz'] = result.x[1]

    return values_dict#, ordered_points[:newPartition]

def compare_points(p1, p2, p3):
    plt.scatter(x=p1[:, [0]], y=p1[:, [1]], label='measured')
    plt.scatter(x=p2[:, [0]], y=p2[:, [1]], label='k-undistorted')
    plt.scatter(x=p3[:, [0]], y=p3[:, [1]], label='calib-undistorted')
    plt.legend()
    plt.show()

def get_calibration_values(input_path, report=False, optimise=True, undistort=True):
    """
    Determine calibration parameters given a set of calibration points and experiment information
    :param input_path: string, path to json file to read
    :param report: bool, whether or not to print details of calibration
    :param optimise: bool, whether or not to optimise the calibration parameters before returning them
    :param undistort: bool, whether or not to take first-order radial distortion into account
    :return: dictionary of calibration parameters and information
    """
    report = report
    # get starting values
    filepath, d_x, d_y, cx, cy, f, known_f, model_name = get_setup_values(input_path)
    # read CSV file to get data
    points = []
    with open(filepath) as file:
        for line in file:
            points.append([float(p) for p in line.split(',')])

    points = np.asarray(points)

    uv_d = points[:, [3, 4]]
    r_d = np.asarray([np.sqrt(np.power(p[0] - cx, 2) + np.power(p[1] - cy, 2)) for p in uv_d])
    r_d = r_d.reshape((len(r_d), 1))

    ordered_points = np.asarray([y for x, y in sorted(zip(r_d, points))])  # sort measured points in order of ascending radial extension
    r_d.sort()

    # cut dataset down to selection of points closest to the centre
    #newPartition = len(points)
    if optimise:
        newPartition = max(len(points) // 4, 7)
        points, r_d = ordered_points[:newPartition], r_d[:newPartition]
    #for p in points:
    #    print(p[3], p[4])

    if report: print("---")
    values_dict = get_calibration_matrix(points, d_x, d_y, cx, cy, f, known_f)
    if report: print("---")
    _, _, K1, K1_cube_error, K1_errors = estimate_k1(points, values_dict)
    values_dict['K1'] = K1

    if optimise:
        #values_dict, newPoints = get_optimised_values(newPoints, r_d, values_dict, report=report)
        values_dict = get_optimised_values(points, r_d, values_dict, report=report)
    """else:
        newPoints = newPoints"""
    if report: print("---")

    if undistort:
        uv_K = apply_K1(points[:, [3, 4]], (cx, cy), values_dict['K1'])
        newPoints = np.hstack((points[:, [0, 1, 2]], uv_K))
    else:
        newPoints = points

    #compare_points(points[:, [3, 4]], newPoints[:, [3, 4]], compute_undistorted(values_dict, points))

    #pixel_error, pixel_errors, estimated_pixels, pixel_errors_per_dimension = get_pixel_error(values_dict['full_transform'], newPoints, silenced=(not report))#, K1=values_dict['K1'], r_d=r_d)
    #pixel_error = compute_pt_diff(newPoints[:, [3, 4]], compute_undistorted(values_dict, points))
    pixel_error, pixel_errors, estimated_pixels, pixel_errors_per_dimension = new_pixel_error(newPoints[:, [3, 4]], compute_undistorted(values_dict, points))
    if report: print("---")
    cube_error, cube_errors, estimated_cube_points, cube_errors_per_dimension = get_cube_error(newPoints, values_dict, silenced=(not report))
    values_dict['pixel_error'] = pixel_error
    values_dict['cube_error'] = cube_error

    if report:
        print("---")
        print("pixel error: {:.2f} pixels\ncube error: {:.2f} mm\nK1: {:.15g}".format(pixel_error, cube_error, values_dict['K1']))
        print("---")
        #print_dict(values_dict)
        print("---")

    #create & save new csv file
    out_lines = ["\n{},{},{},{},{},{},{}, ,{},{},{}, ,{},{},{},{},{}, ,{},{}".format(newPoints[i][0], newPoints[i][1], newPoints[i][2], cube_errors[i], cube_errors_per_dimension[0][i], cube_errors_per_dimension[1][i], cube_errors_per_dimension[2][i], estimated_cube_points[i][0], estimated_cube_points[i][1], estimated_cube_points[i][2], newPoints[i][3], newPoints[i][4], pixel_errors[i], pixel_errors_per_dimension[0][i], pixel_errors_per_dimension[1][i], estimated_pixels[i][0], estimated_pixels[i][1]) for i in range(len(newPoints))]
    with open('output/' + model_name + '_errors-out.csv', 'w') as f:
        f.write("X,Y,Z,cube error,X cube error,Y cube error,Z cube error,,X estimated,Y estimated,Z estimated,,u,v,pixel error,u pixel error,v pixel error,,u estimated,v estimated")
        for line in out_lines:
            f.write(line)

    out_dict = {'full_transformation_matrix':values_dict['full_transform'], 'RT':values_dict['RT'], 'values_dict':values_dict}
    f = open('models/' + model_name + '.pickle', 'wb')
    pickle.dump(out_dict, f)
    f.close()

    return values_dict#, newPoints