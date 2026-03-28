from ultralytics import YOLO
from pathlib import Path

data_path = '/scratch/network/db0197/Adeeb Data/North American Open 2013 Ash-Pil/SQUASHTV_NAOPEN-13_PSA_1-M_2-R2_ASH-PIL.mp4'
data_name = Path(data_path).stem

model = YOLO('/scratch/network/db0197/Pipeline/models/best_player.pt')

results = model.track(
    source=data_path,
    tracker='/scratch/network/db0197/Pipeline/models/custom_botsort.yaml',
    save_conf=True,
    project=data_name,
    save_txt=True,
    persist=True,
    stream = True
)

import detect_player

detect_player.save_results(results, '/scratch/network/db0197/Pipeline/predictions/ash-pil.csv')