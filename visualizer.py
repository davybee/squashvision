import cv2
import numpy as np
import matplotlib.pyplot as plt
import calibration

test_img_path = '/scratch/network/db0197/Pipeline/data/rally6_v2/example_frames/frame094.jpg'
test_vid_path = '/scratch/network/db0197/LabelingProcess/Squash Data/elias_makin_london2025/rallies/rally6_v2.mp4'

def visualize_court_points(points, img_path=test_img_path, labels=None):
    '''
    Draws a set of (u, v) pixel points on an image.
    points : list of (u, v) tuples
    labels : optional list of strings to label each point
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

def visualize_trajectories_on_video(trajectories, segments, P,
                                    vid_path=test_vid_path, out_path='annotated.mp4'):
    '''
    Overlays projected 3D trajectory segments as lines onto a video and saves the result.
    The full segment line is shown for all frames within that segment, then switches
    to the next segment.

    trajectories : list of Nx3 numpy arrays (one per segment, world coords)
    start_stops  : list of (start_frame, end_frame) tuples matching trajectories
    P            : 3x4 projection matrix from calibration
    out_path     : path to write the annotated video
    '''
    cap = cv2.VideoCapture(vid_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    color = (255,  60,  60)

    # Pre-project each segment's full trajectory to a polyline (Nx1x2 for cv2)
    seg_lines = []

    for traj, (start, stop) in zip(trajectories, segments):
        proj_pts = calibration.project_points(P, traj)
        polyline = np.array(proj_pts, dtype=np.int32)  # shape (N, 1, 2)
        seg_lines.append((start, stop, polyline, color))

    def get_active_segment(frame_idx):
        for start, stop, polyline, color in seg_lines:
            if start <= frame_idx <= stop:
                return polyline, color
        return None, None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        polyline, color = get_active_segment(frame_idx)
        if polyline is not None:
            cv2.polylines(frame, [polyline], isClosed=False, color=color, thickness=3)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {out_path} ({frame_idx} frames)")
