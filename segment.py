'''
segment.py

PyTorch port of ai-badminton's MLHitDetector, adapted for squash rally segmentation.

Architecture (mirrors MLHitDetector.create_model):
    flat input (num_consec * feature_per_frame)
    → Reshape (num_consec, feature_per_frame)
    → Bidirectional GRU(64, return_sequences=True)
    → Bidirectional GRU(64, return_sequences=True)
    → GlobalMaxPool over time
    → Linear(num_classes)
    → Softmax

Feature vector per frame (mirrors MLHitDetector.detect_hits):
    ball_xy (2) + player_keypoints (N_kp * 2) + court_corners (8)
    total input = num_consec * feature_per_frame
'''

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


def scale_data(x: np.ndarray) -> np.ndarray:
    '''
    Column-wise min-max scaling, matching MLHitDetector.scale_data.
    Even-indexed columns (x-coords) and odd-indexed (y-coords) are scaled
    independently. Zero values are treated as missing and left unchanged.
    '''
    x = np.array(x, dtype=np.float32)
    eps = 1e-6

    def _scale_cols(x, cols):
        x_ = x[:, cols].copy()
        nonzero = np.abs(x_) >= eps
        if nonzero.any():
            m, M = x_[nonzero].min(), x_[nonzero].max()
            if M > m:
                x_[nonzero] = (x_[nonzero] - m) / (M - m) + 1
        x[:, cols] = x_
        return x

    even = [2 * i for i in range(x.shape[1] // 2)]
    odd  = [2 * i + 1 for i in range(x.shape[1] // 2)]
    x = _scale_cols(x, even)
    x = _scale_cols(x, odd)
    return x


class HitNet(nn.Module):
    '''
    PyTorch equivalent of MLHitDetector.create_model().

    Args:
        feature_dim:  Total flat input size = num_consec * feature_per_frame.
        num_consec:   Number of consecutive frames per window (default 12).
        gru_units:    Hidden units per GRU direction (default 64).
        num_classes:  Output classes (default 3 to match original).
    '''
    def __init__(self,
                 feature_dim: int,
                 num_consec:  int = 12,
                 gru_units:   int = 64,
                 num_classes: int = 3):
        super().__init__()
        self.num_consec        = num_consec
        self.feature_per_frame = feature_dim // num_consec

        self.gru1 = nn.GRU(self.feature_per_frame, gru_units,
                            batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(gru_units * 2, gru_units,
                            batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(gru_units * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, feature_dim)
        x, _ = self.gru1(x.view(x.size(0), self.num_consec, self.feature_per_frame))
        x, _ = self.gru2(x)
        x     = x.max(dim=1).values   # GlobalMaxPool over time
        return torch.softmax(self.fc(x), dim=-1)


class SegmentDetector:
    '''
    Squash event detector. Mirrors the structure of MLHitDetector.

    Args:
        ball_df:       DataFrame with columns X, Y indexed by frame number.
        model_path:    Path to a saved HitNet state dict (.pt file).
        player_df:     Optional DataFrame of player keypoints (one row per frame,
                       columns in x0,y0,x1,y1,... order).
        court_corners: Optional (4, 2) array of court corner pixel coordinates.
        fps:           Frames per second of the source video.
        num_consec:    Window size fed to HitNet (default 12).
        num_classes:   Number of output classes (default 3).
        device:        PyTorch device string.
    '''
    def __init__(self,
                 ball_df,
                 model_path,
                 player_df=None,
                 court_corners=None,
                 fps=50,
                 num_consec=12,
                 num_classes=3,
                 device='cpu'):
        self.ball_df       = ball_df
        self.player_df     = player_df
        self.court_corners = court_corners
        self.fps           = fps
        self.num_consec    = num_consec
        self.device        = device

        feature_per_frame = (2
                             + (player_df.shape[1] if player_df is not None else 0)
                             + (8 if court_corners is not None else 0))
        feature_dim = num_consec * feature_per_frame

        self.model = HitNet(feature_dim, num_consec=num_consec, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def _build_features(self) -> np.ndarray:
        '''
        Build the sliding-window feature matrix, mirroring MLHitDetector.detect_hits().

        For each output row the window of num_consec frames is stacked horizontally:
            [frame_i, ..., frame_i+num_consec-1]
        where each frame contributes: ball_xy + player_keypoints + court_corners.

        Returns:
            (N, num_consec * feature_per_frame) float32 array, scaled via scale_data.
        '''
        L = len(self.ball_df)
        Xb = self.ball_df['X'].to_numpy(dtype=np.float32)
        Yb = self.ball_df['Y'].to_numpy(dtype=np.float32)

        if self.court_corners is not None:
            corners = np.array(self.court_corners, dtype=np.float32).flatten()  # (8,)

        x_list = []
        for i in range(self.num_consec):
            end = L - self.num_consec + i + 1
            x_bird = np.stack([Xb[i:end], Yb[i:end]], axis=1)              # (N, 2)

            parts = [x_bird]
            if self.player_df is not None:
                parts.append(self.player_df.values[i:end].astype(np.float32))
            if self.court_corners is not None:
                parts.append(np.tile(corners, (end - i, 1)))

            x_list.append(np.hstack(parts))

        x_inp = np.hstack(x_list)   # (N, num_consec * feature_per_frame)
        return scale_data(x_inp)

    def detect_events(self) -> np.ndarray:
        '''
        Run HitNet over all windows.

        Returns:
            (N, num_classes) float32 probability array, one row per frame window.
            Frame index of row i corresponds to frame i + num_consec - 1 in ball_df.
        '''
        features = self._build_features()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            return self.model(x).cpu().numpy()


# ── JSON helpers ──────────────────────────────────────────────────────────────

def build_segments_json(markers: list, fps: float,
                        total_frames: int, vid_path) -> dict:
    '''Build the full segments JSON from a list of event markers.'''
    segments = [
        {
            'name':        f'segment_{i + 1}',
            'start_frame': markers[i]['frame'],
            'end_frame':   markers[i + 1]['frame'],
            'start_event': markers[i]['type'],
            'end_event':   markers[i + 1]['type'],
        }
        for i in range(len(markers) - 1)
    ]
    return {
        'markers':    markers,
        'segments':   segments,
        'video_info': {
            'path':         str(vid_path),
            'fps':          fps,
            'total_frames': total_frames,
        },
    }


def save_segments_json(data: dict, out_path) -> None:
    '''Write a segments dict to a JSON file.'''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_segments(json_path) -> list:
    '''
    Load a segments JSON and return (start_frame, end_frame) tuples,
    the format expected by trajectory.predict_segments().
    '''
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [(seg['start_frame'], seg['end_frame']) for seg in data['segments']]
