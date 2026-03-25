import calibration
import cv2

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point


# ── Save / Load ──────────────────────────────────────────────────────────────

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

        if filtered_indices is not None:
            valid = filtered_indices[frame]
        else:
            valid = range(len(result.boxes))

        for i in valid:
            track_id = -1
            if result.boxes.id is not None:
                track_id = int(result.boxes.id[i].cpu().item())

            xyxy = result.boxes.xyxy[i].cpu().numpy()
            conf = float(result.boxes.conf[i].cpu().item())

            if result.keypoints is not None:
                xy = result.keypoints.xy.cpu().numpy()[i]
                kp_conf = result.keypoints.conf.cpu().numpy()[i]
                flat = np.column_stack([xy, kp_conf]).ravel()
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


# ── Court polygon ────────────────────────────────────────────────────────────

def get_court_polygon(avg_court, buffer_meters=1.0):
    """
    Build a buffered court polygon in pixel space covering the FULL court,
    extrapolating side lines from front wall -> T line -> back wall.
    """
    front_left = np.array([avg_court['front_left_x'], avg_court['front_left_y']])
    front_right = np.array([avg_court['front_right_x'], avg_court['front_right_y']])
    mid_left = np.array([avg_court['mid_left_x'], avg_court['mid_left_y']])
    mid_right = np.array([avg_court['mid_right_x'], avg_court['mid_right_y']])

    extend_ratio = 4.31 / 5.44
    back_left = mid_left + extend_ratio * (mid_left - front_left)
    back_right = mid_right + extend_ratio * (mid_right - front_right)

    corners = np.array([
        front_left, front_right, mid_right,
        back_right, back_left, mid_left,
    ], dtype=np.float64)

    court_width_m = 6.4
    mid_width_px = np.linalg.norm(mid_right - mid_left)
    px_per_meter = mid_width_px / court_width_m
    buffer_px = buffer_meters * px_per_meter

    poly = Polygon(corners)
    return poly.buffer(buffer_px)


# ── Court-based filtering ────────────────────────────────────────────────────

def get_foot_position(box, keypoints=None, ankle_conf_threshold=0.3):
    """
    Get foot position from keypoints if available, otherwise bottom-center of box.
    Returns (foot_position, bbox_bottom_center).
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

            if court_poly.contains(Point(foot_pos)) and court_poly.contains(Point(bbox_pos)):
                valid.append(i)

        filtered_indices.append(valid)

    return filtered_indices


# ── Histogram-based ID correction ────────────────────────────────────────────

def _get_torso_crop(frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
    """Crop the middle half of the bounding box (torso region)."""
    x1, y1, x2, y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
    h = y2 - y1
    torso_y1 = y1 + h // 4
    torso_y2 = y1 + 3 * h // 4
    return frame[torso_y1:torso_y2, x1:x2]


def _compute_histogram(crop):
    """Compute a normalized HSV histogram for a torso crop."""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _compare_histograms(hist1, hist2):
    """Correlation similarity in [-1, 1]. Higher = more similar."""
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def fix_player_ids(vid_path, player_csv_path, output_csv_path, ambiguity_margin=0.05):
    """
    Post-process player detections to maintain consistent IDs (1 and 2)
    using torso color histograms captured from the first valid frame.

    Pipeline:
        1. Find first frame with exactly 2 detections -> capture reference histograms
        2. Read video sequentially, match each detection to the closest reference
        3. Use tracker ID as tiebreaker when histogram is ambiguous
        4. Resolve same-frame conflicts (both assigned same player)

    Args:
        vid_path:          Path to source video
        player_csv_path:   Path to player_keypoints.csv (output of detect_player_keypoints)
        output_csv_path:   Path to write the corrected CSV
        ambiguity_margin:  Min difference in similarity scores to trust the histogram.
                           Below this, fall back to tracker ID.
    Returns:
        DataFrame with corrected track_id column (values are 1 or 2 only)
    """
    df = pd.read_csv(player_csv_path)
    cap = cv2.VideoCapture(vid_path)

    # ── 1. Initialize reference histograms from first 2-detection frame ──
    frame_counts = df.groupby('frame').size()
    init_frame = frame_counts[frame_counts == 2].index[0]
    init_rows = df[df['frame'] == init_frame]

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {init_frame}")

    ref_tracker_ids = []
    ref_hists_init = []
    for _, row in init_rows.iterrows():
        crop = _get_torso_crop(frame, row['bbox_x1'], row['bbox_y1'],
                               row['bbox_x2'], row['bbox_y2'])
        ref_hists_init.append(_compute_histogram(crop))
        ref_tracker_ids.append(int(row['track_id']))

    # Map the two original tracker IDs to canonical 1 and 2
    canonical_map = {ref_tracker_ids[0]: 1, ref_tracker_ids[1]: 2}
    ref_hists = {1: ref_hists_init[0], 2: ref_hists_init[1]}

    print(f"Initialized references from frame {init_frame}")
    print(f"  Tracker ID {ref_tracker_ids[0]} -> Player 1")
    print(f"  Tracker ID {ref_tracker_ids[1]} -> Player 2")

    # ── 2. Sequential read + per-detection assignment ────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame_num = -1
    current_frame = None
    new_ids = []

    for _, row in df.iterrows():
        frame_num = int(row['frame'])

        # Advance video to the needed frame (sequential, no seeking)
        while current_frame_num < frame_num:
            ret, current_frame = cap.read()
            current_frame_num += 1
            if not ret:
                current_frame = None
                break

        tracker_id = int(row['track_id'])

        # If we can't read the frame, fall back to tracker mapping
        if current_frame is None:
            new_ids.append(canonical_map.get(tracker_id, 1))
            continue

        crop = _get_torso_crop(current_frame, row['bbox_x1'], row['bbox_y1'],
                               row['bbox_x2'], row['bbox_y2'])
        hist = _compute_histogram(crop)

        sim_1 = _compare_histograms(hist, ref_hists[1])
        sim_2 = _compare_histograms(hist, ref_hists[2])
        margin = abs(sim_1 - sim_2)

        if margin > ambiguity_margin:
            # Clear winner — trust histogram
            assigned = 1 if sim_1 > sim_2 else 2
        elif tracker_id in canonical_map:
            # Ambiguous — fall back to tracker ID
            assigned = canonical_map[tracker_id]
        else:
            # Unknown tracker ID + ambiguous — best guess
            assigned = 1 if sim_1 > sim_2 else 2

        canonical_map[tracker_id] = assigned
        new_ids.append(assigned)

    cap.release()

    # ── 3. Apply corrected IDs ───────────────────────────────────────────
    df['track_id'] = new_ids

    # ── 4. Resolve same-frame conflicts ──────────────────────────────────
    #    If both detections in a frame got the same player ID,
    #    flip the weaker-confidence one to the other player.
    for frame_num, group in df.groupby('frame'):
        if len(group) != 2:
            continue
        ids = group['track_id'].values
        if ids[0] == ids[1]:
            idx = group.index
            confs = group['det_conf'].values
            weaker = idx[0] if confs[0] < confs[1] else idx[1]
            df.at[weaker, 'track_id'] = (2 if ids[0] == 1 else 1)

    df.to_csv(output_csv_path, index=False)

    print(f"\nID correction summary:")
    print(f"  Player 1: {(df['track_id'] == 1).sum()} detections")
    print(f"  Player 2: {(df['track_id'] == 2).sum()} detections")
    print(f"  Saved to {output_csv_path}")

    return df


# ── Main pipeline ────────────────────────────────────────────────────────────

def detect_player_keypoints(model_path, data_path, court_points_csv, save_path,
                            tracker_yaml, save_frames=False, stream=False,
                            ankle_conf_threshold=0.3, buffer_meters=1.0):
    '''
    Detects player keypoints, filters by court bounds, and saves to CSV.

    Args:
        model_path:          Path to trained YOLO pose model
        data_path:           Path to input video
        court_points_csv:    Path to court keypoints CSV
        save_path:           Path to save output CSV
        tracker_yaml:        Path to custom BoT-SORT yaml
        save_frames:         Whether to save annotated frames
        stream:              Whether to use streaming mode
        ankle_conf_threshold: Min ankle confidence for foot position
        buffer_meters:       Court polygon buffer in meters

    Returns:
        results: YOLO results list
    '''
    model = YOLO(model_path)
    data_name = Path(data_path).stem

    results = model.track(
        source=data_path,
        tracker=tracker_yaml,
        save_conf=True,
        project=f'data/{data_name}',
        save=save_frames,
        save_frames=save_frames,
        persist=True
    )

    avg_court = calibration.get_avg_keypoints(court_points_csv)
    filtered_indices = filter_results(results, avg_court,
                                      buffer_meters=buffer_meters,
                                      ankle_conf_threshold=ankle_conf_threshold)
    save_results(results, save_path, filtered_indices)

    return results


def main():
    test_model_path = 'models/best_player.pt'
    test_vid = '/scratch/network/db0197/LabelingProcess/Squash Data/elias_makin_london2025/rallies/rally6_v2.mp4'
    court_points_csv = '/scratch/network/db0197/Pipeline/data/rally6_v2/court_labels/court_keypoints.csv'
    tracker_yaml = '/scratch/network/db0197/Pipeline/models/custom_botsort.yaml'

    save_dir = Path('predictions') / Path(test_vid).stem
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = save_dir / 'player_keypoints_raw.csv'
    fixed_csv = save_dir / 'player_keypoints.csv'

    # Step 1: detect + filter by court bounds
    results = detect_player_keypoints(
        test_model_path, test_vid, court_points_csv, raw_csv,
        tracker_yaml, ankle_conf_threshold=0.0, buffer_meters=1.0
    )

    # Step 2: fix player IDs using histogram matching
    fix_player_ids(test_vid, raw_csv, fixed_csv)

    print(f"\nPipeline complete.")
    print(f"  Raw detections:   {raw_csv}")
    print(f"  Fixed detections: {fixed_csv}")


if __name__ == '__main__':
    main()