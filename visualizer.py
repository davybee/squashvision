import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import calibration

test_img_path = '/scratch/network/db0197/Pipeline/data/rally6_v2/example_frames/frame094.jpg'
test_vid_path = '/scratch/network/db0197/LabelingProcess/Squash Data/elias_makin_london2025/rallies/rally6_v2.mp4'

def visualize_court_points(points, img_path=test_img_path, labels=None):
    '''Draws a set of (u, v) pixel points on an image.

    Args:
        points: List of (u, v) tuples to draw
        img_path: Path to the image to draw on
        labels: Optional list of strings to label each point
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
