import pandas as pd
import numpy as np

def get_court_accuracy(avg, gt):
    '''Computes per-keypoint Euclidean distance in px between avg and gt

    Args:
        avg: Series returned by calibration.get_avg_keypoints() of detected court keypoints
        gt: Series returned by calibration.get_avg_keypoints() of ground truth points

    Returns:
        distances: Series of per-keypoint pixel error, indexed by keypoint name
        mean_dist: Mean Euclidean distance across all keypoints
    '''

    def pairwise(iterable):
        it = iter(iterable)
        return list(zip(it, it))
    
    avg_points = pairwise(avg.values)
    gt_points = pairwise(gt.values)

    KEYPOINT_NAMES = ['front_left', 'mid_left', 'mid_right', 'front_right', 'service_right', 'service_left']
    distances = {}
    for kp, (ax, ay), (gx, gy) in zip(KEYPOINT_NAMES, avg_points, gt_points):
        distances[kp] = np.sqrt((ax - gx)**2 + (ay - gy)**2)

    distances = pd.Series(distances)
    mean_dist = distances.mean()

    return distances, mean_dist

def get_traj_accuracy(traj, detected_gt):
    '''Computes per-frame Euclidean reprojection error in px between a fitted trajectory and detected ball positions.

    Args:
        traj: Nx2 array of projected (u, v) trajectory points
        detected_gt: Nx2 array of detected (u, v) ball positions (ground truth)

    Returns:
        distances: 1D numpy array of per-frame pixel error (length N)
        mean_dist: Mean Euclidean distance across all frames
    '''
    traj = np.array(traj)
    detected_gt = np.array(detected_gt)

    distances = np.sqrt(np.sum((traj - detected_gt) ** 2, axis=1))
    mean_dist = distances.mean()

    return distances, mean_dist

def main():
    import calibration
    import trajectory
    import json

    ## testing the court accuracy metric
    test_gt = 'data/rally6_v2/court_labels/makin_ground_truth.csv'
    test_detection = 'predictions/rally6_v2/court_keypoints.csv'

    avg = calibration.get_avg_keypoints(test_detection)
    gt = calibration.get_avg_keypoints(test_gt, min_vis=0)

    distances, mean_dist = get_court_accuracy(avg, gt)
    print('Mean distance in px', mean_dist)
    print('Keypoint distances:\n', distances, sep='')

    ## testing the trajectory accuracy metric
    test_ball_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/rally6_v2_ball.csv'
    test_segments_json = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/manual_segment.json'

    detected_ball = pd.read_csv(test_ball_csv)
    detected_ball = detected_ball[['X', 'Y']]

    with open(test_segments_json, 'r') as file:
        data = json.load(file)
    segments = [(seg['start_frame'], seg['end_frame']) for seg in data['segments']]

    P = calibration.map_3dcoords(avg)
    FPS = 50
    trajectories = trajectory.predict_segments(detected_ball, segments, P, FPS)
    for i, traj_df in enumerate(trajectories):
        proj_traj = calibration.project_points(P, traj_df.values)
        gt_points = detected_ball.loc[traj_df.index]
        distances, mean_dist = get_traj_accuracy(proj_traj, gt_points.values)
        print(f'Trajectory {i} avg distance:', mean_dist, 'px')

    

if __name__ == '__main__':
    main()