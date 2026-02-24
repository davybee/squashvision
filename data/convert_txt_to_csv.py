import re
from pathlib import Path
import pandas as pd

def convert_yolo_txt_to_csv(labels_dir, output_csv, img_width=None, img_height=None, use_pixels=False):
    """
    Convert YOLO txt predictions to CSV format for squash court keypoints.

    Args:
        labels_dir: Directory containing the YOLO txt label files
        output_csv: Output CSV file path
        img_width: Image width in pixels (required if use_pixels=True)
        img_height: Image height in pixels (required if use_pixels=True)
        use_pixels: If True, convert normalized coords to pixel coords
    """
    if use_pixels and (img_width is None or img_height is None):
        raise ValueError("img_width and img_height must be provided when use_pixels=True")
    labels_path = Path(labels_dir)

    # Get all txt files and sort them by frame number
    txt_files = sorted(labels_path.glob('*.txt'),
                      key=lambda x: int(re.search(r'_(\d+)\.txt$', x.name).group(1)))

    if not txt_files:
        print(f"No txt files found in {labels_dir}")
        return

    # Squash court keypoint labels in order
    keypoint_names = [
        'front_left',
        'mid_left',
        'mid_right',
        'front_right',
        'service_right',
        'service_left'
    ]

    # Prepare CSV header
    # YOLO format: class_id bbox_x bbox_y bbox_w bbox_h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    header = ['frame', 'class_id', 'bbox_x_center', 'bbox_y_center', 'bbox_width', 'bbox_height']

    for kp_name in keypoint_names:
        header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_conf'])

    header.extend(['confidence'])

    # Process all files
    rows = []
    for txt_file in txt_files:
        # Extract frame number from filename
        match = re.search(r'_(\d+)\.txt$', txt_file.name)
        if not match:
            continue
        frame_num = int(match.group(1))

        # Read the txt file
        with open(txt_file, 'r') as f:
            content = f.read().strip()

            if not content:
                # Empty file, skip
                continue

            # Parse YOLO format: class_id bbox_x bbox_y bbox_w bbox_h kp1_x kp1_y kp1_v ...
            values = content.split()
            class_id = int(values[0])

            # Extract bounding box (next 4 values)
            bbox = [float(v) for v in values[1:5]]

            # Extract keypoints (remaining values)
            keypoints = [float(v) for v in values[5:]]

            # Convert to pixel coordinates if requested
            if use_pixels:
                # Convert bbox: [x_center, y_center, width, height]
                bbox[0] *= img_width   # x_center
                bbox[1] *= img_height  # y_center
                bbox[2] *= img_width   # width
                bbox[3] *= img_height  # height

                # Convert keypoints: [x, y, conf, x, y, conf, ...]
                for i in range(0, len(keypoints), 3):
                    if i + 1 < len(keypoints):  # Ensure we have both x and y
                        keypoints[i] *= img_width      # x coordinate
                        keypoints[i+1] *= img_height   # y coordinate
                        # keypoints[i+2] is confidence, leave as-is

            # Create row: frame, class_id, bbox values, keypoint values
            row = [frame_num, class_id] + bbox + keypoints
            rows.append(row)

    # Write to CSV using pandas (overwrites if file exists)
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(output_csv, index=False)

    print(f"Converted {len(rows)} frames from {labels_dir}")
    print(f"Output saved to {output_csv}")
    print(f"CSV contains {len(header)} columns")
    print(f"Keypoints: {', '.join(keypoint_names)}")
    if use_pixels:
        print(f"Coordinates converted to pixels (image: {img_width}x{img_height})")
    else:
        print(f"Coordinates in normalized YOLO format (0-1 range)")

if __name__ == '__main__':
    # Configuration
    labels_directory = '/scratch/network/db0197/Pipeline/data/rally6_v2/court_labels/labels'
    output_csv_file = '/scratch/network/db0197/Pipeline/data/rally6_v2/court_labels/court_predictions.csv'

    # Image dimensions
    IMAGE_WIDTH = 1920  
    IMAGE_HEIGHT = 1080 

    # Convert - set use_pixels=True to convert to pixel coordinates
    convert_yolo_txt_to_csv(
        labels_directory,
        output_csv_file,
        img_width=IMAGE_WIDTH,
        img_height=IMAGE_HEIGHT,
        use_pixels=True  # Set to False for normalized YOLO format
    )
