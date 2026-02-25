import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
import calibration

def create_3d_trajectory(params, times):
    '''
    Creates a trajectory of 3D world points given initial conditions. 

    ** Currently Implemented with a basic newton + drag equation **

    Args:
        params: The intial position (x,y,z) and initial velocity (vx0, vy0, vz0) of the ball. Each is an element of a list. 

    Returns:
        Times x 4 list of 3D homogenous coords in real world space based on intial conditions.
    '''

    x0, y0, z0, vx0, vy0, vz0, drag = params
    g = 9.81

    n_points = len(times)
    positions = np.zeros((n_points, 4))

    for i, t in enumerate(times):
        positions[i, 0] = x0 + vx0 * t * np.exp(-drag * t)
        positions[i, 1] = y0 + vy0 * t * np.exp(-drag * t)
        positions[i, 2] = z0 + vz0 * t - 0.5 * g * t ** 2
        positions[i, 3] = 1 # homogenous coords

    return positions

def relu(x):
    return np.maximum(0.0, x)

def court_violation_residuals(points_3d, weight=10.0):
    '''
    Computes soft constraint residuals for squash court boundary violations, per frame.

    Coordinate system:
    x: width,  -3.2 (left wall)  to  3.2 (right wall)
    y: depth,  -4.31 (back wall) to  5.44 (front wall), with 0 at center of T
    z: height,  0 (floor)        to 10m max

    Args:
        points_3d: (N x 4) array of homogeneous 3D world coordinates.
        weight: scalar controlling strength of boundary enforcement.

    Returns:
        residuals: 1D numpy array of boundary violation residuals,
                   scaled for use in least squares optimization.
    '''
    x_min, x_max = -3.2, 3.2
    y_min, y_max = -4.31, 5.44
    z_min, z_max = 0.0, 10.0

    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    vx_hi = relu(x - x_max)
    vx_lo = relu(x_min - x)
    vy_hi = relu(y - y_max)
    vy_lo = relu(y_min - y)
    vz_hi = relu(z - z_max)
    vz_lo = relu(z_min - z)

    w = np.sqrt(weight)
    return w * np.concatenate([vx_hi, vx_lo, vy_hi, vy_lo, vz_hi, vz_lo], axis=0)

def compute_loss(points_2d, projected_points):
    '''MSE reprojection error between ground truth and projected 2D points.'''
    diff = points_2d - projected_points
    return np.mean(diff ** 2)

def predict_3d_trajectory(seg_points_2d, start, stop, P, FPS):
    '''
    Optimizes 3D trajectory params to minimize reprojection error against 2D (u, v) ground truth.

    Optimizes with per frame residuals
    Args: 
        seg_points_2d: Nx2 DataFrame of (u, v) pixel coordinates for this segment
        P: 3x4 projection matrix from calibration
    Returns: 
        Nx3 array of optimized 3D positions
    '''
    times = np.arange(stop - start + 1, dtype=np.float64) / FPS
    detected_points = seg_points_2d.values.astype(np.float64)  # Nx2

    def residuals(params):
        points_3d = create_3d_trajectory(params, times)              # Nx4
        proj_points = calibration.project_points(P, points_3d)       # Nx2

        reproj_res = (proj_points - detected_points).reshape(-1)     # (2N,)

        court_res = court_violation_residuals(points_3d, weight=10.0) # tune weight if needed
        return np.concatenate([reproj_res, court_res], axis=0)

    x0 = np.array([0.0, 0.0, 1.0,
                   0.0, 0.0, 0.0,
                   0.1], dtype=np.float64)

    lb = np.array([-3.2, -4.31, 0.0,
                   -30.0, -30.0, -20.0,
                   0.0], dtype=np.float64)
    ub = np.array([ 3.2,  5.44, 10.0,
                    30.0,  30.0,  20.0,
                    5.0], dtype=np.float64)

    result = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=3.0,
        x_scale="jac",
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=20000
    )

    return create_3d_trajectory(result.x, times)

def predict_segments(detected_points, segments, P, FPS):
    '''
    Predicts the 3D points that best optimize the loss function per segment.
    Args:
        detected_points: A dataframe containing the X and Y detected ball points in (u, v) coordinates. our "ground truth"
        segments: A list of tuples of the form (start frame, end frame) for the individual segments
        P: the projection matrix that takes points from 3D to 2D
        FPS: The frames per second of the video being detected

    Returns:
        A list of DataFrames, one per segment, with frame number as index and columns ['x', 'y', 'z', 'w']
        representing 3D homogeneous coords in meters, with (0,0,0) at the center of the T.
    '''
    trajectories = []
    for start, stop in segments:
        seg_points_2d = detected_points.iloc[start:stop + 1]
        traj = predict_3d_trajectory(seg_points_2d, start, stop, P, FPS)  # Nx4 numpy array

        traj_df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'w'], index=range(start, stop + 1))
        trajectories.append(traj_df)

        proj_traj = calibration.project_points(P, traj)
        loss = compute_loss(seg_points_2d.values, proj_traj)
        print(f"Segment {start}-{stop}: {len(traj)} points, first={traj[0]}, last={traj[-1]}\nLoss: {loss}")

    return trajectories

def main():
    test_court_csv = '/scratch/network/db0197/Pipeline/predictions/rally6_v2/court_keypoints.csv'
    test_ball_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/rally6_v2_ball.csv'
    test_segments_json = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/manual_segment.json'
    FPS = 50
    
    detected_points = pd.read_csv(test_ball_csv)
    detected_points = detected_points[['X', 'Y']]

    avg = calibration.get_avg_keypoints(test_court_csv)
    P = calibration.map_3dcoords(avg)

    with open(test_segments_json, 'r') as file:
        data = json.load(file)
    segments = [(seg['start_frame'], seg['end_frame']) for seg in data['segments']]

    trajectories = predict_segments(detected_points, segments, P, FPS)
    print(trajectories[0])
    
if __name__ == '__main__':
    main()