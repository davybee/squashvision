import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import calibration
import pandas as pd

test_img_path = '/scratch/network/db0197/Pipeline/data/rally6_v2/frames/frame0094.jpg'
test_vid_path = '/scratch/network/db0197/LabelingProcess/Squash Data/elias_makin_london2025/rallies/rally6_v2.mp4'

def visualize_court_points(points, img_path=test_img_path, labels=None, save_path=None):
    '''Draws a set of (u, v) pixel points on an image.

    Args:
        points: List of (u, v) tuples to draw
        img_path: Path to the image to draw on
        labels: Optional list of strings to label each point
        save_path: Optional path to save the figure (e.g. 'out.png'). If None, displays interactively.
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i, (u, v) in enumerate(points):
        u, v = int(round(u)), int(round(v))
        cv2.circle(img, (u, v), radius=8, color=(255, 0, 0), thickness=-1)
        if labels is not None:
            cv2.putText(img, labels[i], (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()




def visualize_trajectories_on_video(trajectories, P, ball_df,
                                    vid_path=test_vid_path, out_path='annotated.mp4',
                                    show_ball=True):
    '''Overlays projected trajectory polylines and per-frame ball detections onto a video.

    Args:
        trajectories: List of DataFrames (one per segment), with frame number as index
            and columns ['x', 'y', 'z', 'w'] in world coords
        P: 3x4 projection matrix from calibration
        ball_df: DataFrame of ball detections indexed by frame, with columns ['X', 'Y']
        vid_path: Path to the source video
        out_path: Path to write the annotated output video
        show_detections: If True, draws the raw ball detection dot on each frame
    '''

    cap = cv2.VideoCapture(vid_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Pre-project each segment's full trajectory to a polyline
    seg_lines = []
    for traj_df in trajectories:
        start, stop = traj_df.index[0], traj_df.index[-1]
        proj_pts = calibration.project_points(P, traj_df.values)
        polyline = np.array(proj_pts, dtype=np.int32)
        seg_lines.append((start, stop, polyline))

    def get_active_segment(frame_idx):
        for start, stop, polyline in seg_lines:
            if start <= frame_idx <= stop:
                return polyline
        return None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        polyline = get_active_segment(frame_idx)
        if polyline is not None:
            cv2.polylines(frame, [polyline], isClosed=False, color=(60, 60, 255), thickness=3)

            if show_ball and frame_idx in ball_df.index:
                u, v = int(ball_df.at[frame_idx, 'X']), int(ball_df.at[frame_idx, 'Y'])
                cv2.circle(frame, (u, v), radius=6, color=(0, 255, 0), thickness=-1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {out_path} ({frame_idx} frames)")

def visualize_traj_vs_detections(trajectories, P, ball_df, out_dir, width=1920, height=1080):
    '''Saves one image per trajectory segment showing the projected trajectory and ball detections on a black canvas.

    Args:
        trajectories: List of DataFrames (one per segment), with frame number as index
            and columns ['x', 'y', 'z', 'w'] in world coords
        P: 3x4 projection matrix from calibration
        ball_df: DataFrame of ball detections indexed by frame, with columns ['X', 'Y']
        width: Canvas width in pixels, should match source video
        height: Canvas height in pixels, should match source video

    Returns:
        Saves images to predictions/traj_vs_detections/traj_XXXX.jpg
    '''
    
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, traj_df in enumerate(trajectories):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        proj_pts = calibration.project_points(P, traj_df.values)
        polyline = np.array(proj_pts, dtype=np.int32)
        cv2.polylines(canvas, [polyline], isClosed=False, color=(60, 60, 255), thickness=3)

        seg_detections = ball_df.loc[ball_df.index.intersection(traj_df.index)]
        for _, row in seg_detections.iterrows():
            u, v = int(row['X']), int(row['Y'])
            cv2.circle(canvas, (u, v), radius=6, color=(0, 255, 0), thickness=-1)

        out_path = out_dir / f'traj_{i + 1:04d}.jpg'
        cv2.imwrite(str(out_path), canvas)

    print(f"Saved {len(trajectories)} images to {out_dir}/")

def visualize_ball_detections(vid_path, ball_df, out_path):
    cap = cv2.VideoCapture(vid_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # csv indexed at 1 right now.
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in ball_df.index:
            u, v = int(ball_df.at[frame_idx, 'X']), int(ball_df.at[frame_idx, 'Y'])
            cv2.circle(frame, (u, v), radius=6, color=(0, 255, 0), thickness=-1)
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {out_path} ({frame_idx} frames)")

def visualize_players(vid_path, player_df, out_path, kp_conf_threshold=0.0, det_conf_threshold=0.0):
    '''Overlays player keypoints, skeletons, bounding boxes, and track IDs onto a video.
    Args:
        vid_path: Path to the source video
        player_df: DataFrame from player_keypoints.csv (columns: frame, track_id, bbox_*, kp_*)
        out_path: Path to write the annotated output video
        kp_conf_threshold: Minimum keypoint confidence to draw (default 0.3)
        det_conf_threshold: Minimum detection confidence to show a player at all (default 0.0)
    '''
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    PLAYER_COLORS = [(0, 200, 255), (255, 100, 0), (0, 255, 100), (200, 0, 255)]
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    ]
    kp_x_cols = [f'{kp}_x' for kp in KEYPOINT_NAMES]
    kp_y_cols = [f'{kp}_y' for kp in KEYPOINT_NAMES]
    kp_v_cols = [f'{kp}_vis' for kp in KEYPOINT_NAMES]

    frame_groups = player_df.groupby('frame')

    cap = cv2.VideoCapture(vid_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_groups.groups:
            for _, row in frame_groups.get_group(frame_idx).iterrows():
                track_id = int(row['track_id'])
                det_conf = float(row['det_conf'])
                color = PLAYER_COLORS[track_id % len(PLAYER_COLORS)]

                # Bounding box
                x1, y1, x2, y2 = int(row['bbox_x1']), int(row['bbox_y1']), int(row['bbox_x2']), int(row['bbox_y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label with ID and detection confidence
                label = f'ID {track_id} ({det_conf:.2f})'
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Keypoints with per-keypoint confidence
                xs = row[kp_x_cols].values.astype(float)
                ys = row[kp_y_cols].values.astype(float)
                vs = row[kp_v_cols].values.astype(float)

                for kx, ky, kv in zip(xs, ys, vs):
                    if kv > kp_conf_threshold:
                        cv2.circle(frame, (int(kx), int(ky)), 4, color, -1)
                        cv2.putText(frame, f'{kv:.1f}', (int(kx) + 5, int(ky) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, kp_conf_threshold, color, 1)

                # Skeleton
                for a, b in SKELETON:
                    if vs[a] > kp_conf_threshold and vs[b] > kp_conf_threshold:
                        cv2.line(frame, (int(xs[a]), int(ys[a])),
                                 (int(xs[b]), int(ys[b])), color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {out_path} ({frame_idx} frames)")

def main():
    # ball_df_path = '/scratch/network/db0197/Pipeline/data/rally6_v2/ball_labels/rally6_v2_ball.csv'
    # ball_df = pd.read_csv(ball_df_path)
    # ball_df = ball_df[['X', 'Y']]
    # out_path_ball = 'predictions/rally6_v2/annotated_ball.mp4'

    court_path = '/scratch/network/db0197/Pipeline/data/rally6_v2/court_labels/court_keypoints.csv'

    import calibration
    avg = calibration.get_avg_keypoints(court_path)
    points = calibration.pairwise(avg.values)

    visualize_court_points(points, save_path='/scratch/network/db0197/Pipeline/predictions/rally6_v2')

    out_path_players = 'predictions/rally6_v2/annotated_players_swap_raw.mp4'
    player_df = pd.read_csv('/scratch/network/db0197/Pipeline/predictions/rally6_v2/player_keypoints.csv')
    visualize_players(test_vid_path, player_df, out_path_players)

    out_path_players = 'predictions/rally6_v2/annotated_players_swap_try.mp4'
    player_df = pd.read_csv('/scratch/network/db0197/Pipeline/predictions/rally6_v2/player_keypoints_raw.csv')
    visualize_players(test_vid_path, player_df, out_path_players)

    
    # out_path_players = 'predictions/rally6_v2/annotated_players_unedited_track.mp4'
    # player_df = pd.read_csv('/scratch/network/db0197/Pipeline/predictions/rally6_v2/player_keypoints_unedited_track.csv')
    # visualize_players(test_vid_path, player_df,out_path_players)

    # out_path_players = 'predictions/rally6_v2/annotated_players_unedited_predict.mp4'
    # player_df = pd.read_csv('/scratch/network/db0197/Pipeline/predictions/rally6_v2/player_keypoints_unedited_predict.csv')
    # visualize_players(test_vid_path, player_df,out_path_players)

    # visualize_ball_detections(test_vid_path, ball_df, out_path)
    

if __name__ == '__main__':
    main()