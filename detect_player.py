import calibration

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np
import numpy as np
from shapely.geometry import Polygon, Point

def save_results(results, save_path, filtered_indices=None):
    '''
    Saves filtered results into a csv.
    Each row is one detection (frame + player).
    If filtered_indices is None, saves all detections.
    '''
    KEYPOINT_NAMES = [
        'nose',
        'left_eye', 'right_eye',
        'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
    ]
    KP_COLS = [f'{kp}_{s}' for kp in KEYPOINT_NAMES for s in ('x', 'y', 'vis')]
    COLS = ['frame', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'det_conf'] + KP_COLS

    rows = []
    for frame, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Determine which detections to save
        if filtered_indices is not None:
            valid = filtered_indices[frame]
        else:
            valid = range(len(result.boxes))

        for i in valid:
            # Track ID (may be None if tracking lost)
            track_id = -1
            if result.boxes.id is not None:
                track_id = int(result.boxes.id[i].cpu().item())

            # Bounding box and confidence
            xyxy = result.boxes.xyxy[i].cpu().numpy()
            conf = float(result.boxes.conf[i].cpu().item())

            # Keypoints
            if result.keypoints is not None:
                xy = result.keypoints.xy.cpu().numpy()[i]      # (17, 2)
                kp_conf = result.keypoints.conf.cpu().numpy()[i]  # (17,)
                flat = np.column_stack([xy, kp_conf]).ravel()  # (51,)
            else:
                flat = np.zeros(len(KP_COLS))

            row = np.concatenate([
                [frame, track_id, xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf],
                flat
            ])
            rows.append(row)

    df = pd.DataFrame(np.array(rows), columns=COLS)
    df['frame'] = df['frame'].astype(int)
    df['track_id'] = df['track_id'].astype(int)
    df.to_csv(save_path, index=False)

def get_court_polygon(avg_court, buffer_meters=1.0):
    """
    Build a buffered court polygon in pixel space covering the FULL court,
    extrapolating side lines from front wall → T line → back wall.
    """
    front_left = np.array([avg_court['front_left_x'], avg_court['front_left_y']])
    front_right = np.array([avg_court['front_right_x'], avg_court['front_right_y']])
    mid_left = np.array([avg_court['mid_left_x'], avg_court['mid_left_y']])
    mid_right = np.array([avg_court['mid_right_x'], avg_court['mid_right_y']])

    # Front wall to T = 5.44m, T to back wall = 4.31m
    # Extrapolate side lines past T toward back wall
    extend_ratio = 4.31 / 5.44
    back_left = mid_left + extend_ratio * (mid_left - front_left)
    back_right = mid_right + extend_ratio * (mid_right - front_right)

    corners = np.array([
        front_left,
        front_right,
        mid_right,
        back_right,
        back_left,
        mid_left,
    ], dtype=np.float64)

    court_width_m = 6.4
    mid_width_px = np.linalg.norm(mid_right - mid_left)
    px_per_meter = mid_width_px / court_width_m
    buffer_px = buffer_meters * px_per_meter

    poly = Polygon(corners)
    buffered_poly = poly.buffer(buffer_px)

    return buffered_poly

def get_foot_position(box, keypoints=None, ankle_conf_threshold=0.3):
    """
    Get foot position from keypoints if available, otherwise bottom-center of box.
    Also returns bbox bottom-center for validation.
    """
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16

    xyxy = box.xyxy.cpu().numpy()[0]
    bbox_bottom_center = np.array([(xyxy[0] + xyxy[2]) / 2, xyxy[3]])

    if keypoints is not None:
        kpts = keypoints.data.cpu().numpy()[0]
        left = kpts[ANKLE_LEFT]
        right = kpts[ANKLE_RIGHT]

        left_valid = left[2] > ankle_conf_threshold
        right_valid = right[2] > ankle_conf_threshold
        
        if left_valid and right_valid:
            foot = (left[:2] + right[:2]) / 2
        elif left_valid:
            foot = left[:2]
        elif right_valid:
            foot = right[:2]
        else:
            foot = bbox_bottom_center
        
        return foot, bbox_bottom_center
    
    return bbox_bottom_center, bbox_bottom_center


def filter_results(results, avg_court, buffer_meters=1.0, ankle_conf_threshold=0.3):
    """
    Filters out detections where EITHER the feet OR the bounding box
    are outside the court + buffer.
    """
    court_poly = get_court_polygon(avg_court, buffer_meters=buffer_meters)

    filtered_indices = []

    for result in results:
        valid = []

        if result.boxes is None or len(result.boxes) == 0:
            filtered_indices.append(valid)
            continue

        for i in range(len(result.boxes)):
            kpts = result.keypoints[i] if result.keypoints is not None else None
            foot_pos, bbox_pos = get_foot_position(result.boxes[i], kpts, ankle_conf_threshold)
            
            # BOTH foot and bbox must be inside the court polygon
            if court_poly.contains(Point(foot_pos)) and court_poly.contains(Point(bbox_pos)):
                valid.append(i)
        
        filtered_indices.append(valid)
    
    return filtered_indices

def detect_player_keypoints(model_path, data_path, court_points_csv, save_path, save_frames=False, stream=False, ankle_conf_threshold=0.3):
    ''' Detects the player keypoints and saves them to a csv.

    Args:
        model_path: The path to your trained pose model
        data_path: The path to the data being detectedd

    Returns:
        Returns the results in memory if stream=False. Returns nothing if stream=True
    '''
    model = YOLO(model_path)
    data_name = Path(data_path).stem

    results = model.track(
        source=data_path,
        tracker="/scratch/network/db0197/Pipeline/models/custom_botsort.yaml",
        save_conf=True,
        project=f'data/{data_name}', 
        save=save_frames,
        save_frames=save_frames,
        stream=stream,
        persist=True
    )

    # results2 = model.predict(
    #     source=data_path,
    #     save_conf=True,
    #     project=f'data/{data_name}', 
    #     save=save_frames,
    #     save_frames=save_frames,
    #     stream=stream
    # )

    ## testing filtering
    # test_save_path = 'predictions/rally6_v2/player_keypoints_unedited_track.csv'
    # save_results(results, test_save_path)

    # test_save_path = 'predictions/rally6_v2/player_keypoints_unedited_predict.csv'
    # save_results(results2, test_save_path)

    avg_court = calibration.get_avg_keypoints(court_points_csv)
    filtered_indices = filter_results(results, avg_court, buffer_meters=1.0, ankle_conf_threshold=0)
    save_path = save_path = 'predictions/rally6_v2/player_keypoints_0.csv'
    save_results(results, save_path, filtered_indices)

    filtered_indices = filter_results(results, avg_court, buffer_meters=1.0, ankle_conf_threshold=0.1)
    save_path = save_path = 'predictions/rally6_v2/player_keypoints_1.csv'
    save_results(results, save_path, filtered_indices)

    filtered_indices = filter_results(results, avg_court, buffer_meters=1.0, ankle_conf_threshold=0.2)
    save_path = save_path = 'predictions/rally6_v2/player_keypoints_2.csv'
    save_results(results, save_path, filtered_indices)

    return results

def main():
    test_model_path = 'models/best_player.pt'
    test_vid = '/scratch/network/db0197/LabelingProcess/Squash Data/elias_makin_london2025/rallies/rally6_v2.mp4'
    court_points_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/court_labels/court_keypoints.csv'

    # saves to predictions/{vid_name}/player_keypoints.csv
    save_path_parents = Path('predictions') / Path(test_vid).stem
    save_path_parents.mkdir(parents=True, exist_ok=True)
    save_path = save_path_parents / 'player_keypoints'

    results = detect_player_keypoints(test_model_path, test_vid, court_points_csv, save_path, ankle_conf_threshold=0.0)

    print(type(results))

if __name__ == '__main__':
    main()