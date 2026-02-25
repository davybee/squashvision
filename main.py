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

def pipeline(vid_path, save_path_parents, player_model, court_model, segs=None, visualize=False, FPS=50, width=1920, height=1080):
    ### need to automate ###
    # FPS=50
    #detected_points = detect_ball.detect()
    #detected_segments = detect_segments.detect()
    save_path_player = save_path_parents / 'player_keypoints.csv'
    detect_player.detect_player_keypoints(player_model, vid_path, save_path_player)

    save_path_court = save_path_parents / 'court_keypoints.csv'
    detect_court.detect_court_keypoints(court_model, vid_path, save_path_court)
    
    # get projection matrix
    court_csv = f'predictions/{vid_path.stem}/court_keypoints.csv'
    avg_points = calibration.get_avg_keypoints(court_csv)
    P = calibration.map_3dcoords(avg_points)

    # get player points
    player_csv = f'predictions/{vid_path.stem}/player_keypoints.csv'
    player_df = pd.read_csv(player_csv)

    # get ball points
    ## NEED TO ADJUST FOR AUTO ##
    ball_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/rally6_v2_ball.csv'
    ball_df = pd.read_csv(ball_csv)[['X', 'Y']]

    ## NEED TO GET AUTO SEGMENTS ##
    if segs is not None:
        segments = segs
    
    # get trajectories
    trajectories = trajectory.predict_segments(ball_df, segments, P, FPS)
    visualizer.visualize_trajectories_on_video(trajectories, P, ball_df, vid_path=vid_path, out_path='predictions/rally6_v2/annotated_with_ball.mp4', show_ball=True)
    visualizer.visualize_trajectories_on_video(trajectories, P, ball_df, vid_path=vid_path, out_path='predictions/rally6_v2/annotated_without_ball.mp4', show_ball=False)
    visualizer.visualize_traj_vs_detections(trajectories, P, ball_df, width=width, height=height)

    print('Done!')


def test():
    test_segments_json = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/manual_segment.json'
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
             height=HEIGHT)



if __name__ == '__main__':
    test()