import pandas as pd
import json
from pathlib import Path
import cv2

import calibration
import visualizer
import trajectory
import detect_ball
import detect_court
import detect_player

def _resolve_run_dir(base_dir: Path, run_name: str = None) -> Path:
    '''
    Resolve the run subdirectory under base_dir.
    If run_name is given, use it (overwrite if exists).
    Otherwise, auto-generate the next runN that does not exist.
    '''
    if run_name is not None:
        run_dir = base_dir / run_name
    else:
        n = 1
        while (base_dir / f'run{n}').exists():
            n += 1
        run_dir = base_dir / f'run{n}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def pipeline(vid_path, save_path_parents, player_model, court_model, segs=None, visualize=False, FPS=50, width=1920, height=1080, run_name=None):
    ### need to automate ###
    # FPS=50
    #detected_points = detect_ball.detect()
    #detected_segments = detect_segments.detect()
    run_dir = _resolve_run_dir(save_path_parents, run_name)
    print(f'Saving to: {run_dir}')

    save_path_player = run_dir / 'player_keypoints.csv'
    detect_player.detect_player_keypoints(player_model, vid_path, save_path_player)

    save_path_court = run_dir / 'court_keypoints.csv'
    detect_court.detect_court_keypoints(court_model, vid_path, save_path_court)

    # get projection matrix
    avg_points = calibration.get_avg_keypoints(str(save_path_court))
    P = calibration.map_3dcoords(avg_points)

    # get player points
    player_df = pd.read_csv(save_path_player)

    # get ball points
    ## NEED TO ADJUST FOR AUTO ##
    ball_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/rally6_v2_ball_corrected_pixels.csv'
    ball_df = pd.read_csv(ball_csv)[['X', 'Y']]

    ## NEED TO GET AUTO SEGMENTS ##
    if segs is not None:
        segments = segs

    # get trajectories
    trajectories = trajectory.predict_segments(ball_df, segments, P, FPS)

    # save 3D ball coordinates (one row per frame, NaN for frames outside any segment)
    ball_3d = pd.concat(trajectories)
    ball_3d.index.name = 'frame'
    ball_3d.to_csv(run_dir / 'ball_3d.csv')

    visualizer.visualize_trajectories_on_video(trajectories, P, ball_df, vid_path=vid_path, out_path=str(run_dir / 'annotated_with_ball.mp4'), show_ball=True)
    visualizer.visualize_trajectories_on_video(trajectories, P, ball_df, vid_path=vid_path, out_path=str(run_dir / 'annotated_without_ball.mp4'), show_ball=False)

    out_dir = run_dir / 'traj_vs_detections'
    visualizer.visualize_traj_vs_detections(trajectories, P, ball_df, out_dir, width=width, height=height)

    print('Done!')

def test_all(run_name=None):
    test_segments_json = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/manual_segment_new.json'
    test_vid=Path('/scratch/network/db0197/LabelingProcess/Squash Data/elias_makin_london2025/rallies/rally6_v2.mp4')

    player_model = '/scratch/network/db0197/Pipeline/models/best_player.pt'
    court_model = '/scratch/network/db0197/Pipeline/models/best_court.pt'

    save_path_parents = Path('predictions') / Path(test_vid).stem
    save_path_parents.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(test_vid))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(test_segments_json, 'r') as file:
        data = json.load(file)
        segments = [(seg['start_frame'], seg['end_frame']) for seg in data['segments']]

    pipeline(vid_path=test_vid,
             save_path_parents=save_path_parents,
             player_model=player_model,
             court_model=court_model,
             segs=segments,
             visualize=True,
             FPS=FPS,
             width=WIDTH,
             height=HEIGHT,
             run_name=run_name)

def pipeline_optimization(vid_path, save_path_parents, court_csv, ball_csv, segs=None, FPS=50, width=1920, height=1080, run_name=None):
    run_dir = _resolve_run_dir(save_path_parents, run_name)
    print(f'Saving to: {run_dir}')

    # get projection matrix from pre-calculated court positions
    avg_points = calibration.get_avg_keypoints(court_csv)
    P = calibration.map_3dcoords(avg_points)

    # get ball points
    ball_df = pd.read_csv(ball_csv)[['X', 'Y']]

    ## NEED TO GET AUTO SEGMENTS ##
    if segs is not None:
        segments = segs

    # get trajectories
    trajectories = trajectory.predict_segments(ball_df, segments, P, FPS)

    # save 3D ball coordinates (one row per frame, NaN for frames outside any segment)
    ball_3d = pd.concat(trajectories)
    ball_3d.index.name = 'frame'
    ball_3d.to_csv(run_dir / 'ball_3d.csv')

    visualizer.visualize_trajectories_on_video(trajectories, P, ball_df, vid_path=vid_path, out_path=str(run_dir / 'annotated_with_ball.mp4'), show_ball=True)
    visualizer.visualize_trajectories_on_video(trajectories, P, ball_df, vid_path=vid_path, out_path=str(run_dir / 'annotated_without_ball.mp4'), show_ball=False)

    out_dir = run_dir / 'traj_vs_detections'
    visualizer.visualize_traj_vs_detections(trajectories, P, ball_df, out_dir, width=width, height=height)

    print('Done!')


def test_optimization(run_name=None):
    test_segments_json = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/manual_segment_new.json'
    test_vid = Path('/scratch/network/db0197/LabelingProcess/Squash Data/elias_makin_london2025/rallies/rally6_v2.mp4')
    court_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/court_labels/court_keypoints.csv'

    # ball_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/rally6_v2_ball_corrected_pixels.csv'
    ball_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/rally6_v2_ball_untouched.csv'

    save_path_parents = Path('predictions') / Path(test_vid).stem
    save_path_parents.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(test_vid))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(test_segments_json, 'r') as file:
        data = json.load(file)
        segments = [(seg['start_frame'], seg['end_frame']) for seg in data['segments']]

    pipeline_optimization(vid_path=test_vid,
                          save_path_parents=save_path_parents,
                          court_csv=court_csv,
                          ball_csv=ball_csv,
                          segs=segments,
                          FPS=FPS,
                          width=WIDTH,
                          height=HEIGHT,
                          run_name=run_name)


if __name__ == '__main__':
    folder = 'linear_bad_ball_run'
    test_optimization(run_name = folder)

    # test_all(run_name = folder)