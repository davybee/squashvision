from ultralytics import YOLO
from pathlib import Path

data_path = '/scratch/network/db0197/Adeeb Data/Canary Wharf 2013/SQUASHTV_CANARYWHARF-13_PSA_1-M_2-QF_MoSHO-MUS.mp4'
data_name = Path(data_path).stem

model = YOLO('/scratch/network/db0197/Pipeline/models/best_player.pt')

results = model.track(
    source=data_path,
    tracker='/scratch/network/db0197/Pipeline/models/custom_botsort.yaml',
    save_conf=True,
    project=data_name,
    save=True,
    save_txt=True,
    persist=True,
    stream = True
)

for r in results:
    r