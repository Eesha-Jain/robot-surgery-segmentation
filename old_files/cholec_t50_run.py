import cv2
import os
import random
import json

def draw_bounding_boxes(img_dir, labels, output_path):
    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get list of PNG files in the directory
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    for frame_file in frame_files:
        # Extract the frame index from the file name
        frame_index = int(frame_file.split('.')[0])  # Assuming files are named like '000001.png'

        # Read the image
        frame_path = os.path.join(img_dir, frame_file)
        frame = cv2.imread(frame_path)

        # Get labels for the current frame
        frame_labels = labels.get(str(frame_index), None)

        if frame_labels:
            height, width, _ = frame.shape  # Get image dimensions

            for label in frame_labels:
                # Extracting bounding box information
                tool_id = label[0]  # ID for the triplet
                x1 = label[3] * width  # Scaled x1 coordinate
                y1 = label[4] * height  # Scaled y1 coordinate
                bw = label[5] * width  # Scaled bounding box width
                bh = label[6] * height  # Scaled bounding box height

                # Calculate the bottom-right coordinates
                x2 = x1 + bw
                y2 = y1 + bh

                # Check if coordinates are valid
                if all(coord != -1 for coord in [x1, y1, bw, bh]):
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Tool {tool_id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the frame with bounding boxes
        cv2.imwrite(os.path.join(output_path, frame_file), frame)

    print(f"Video frames with bounding boxes saved to {output_path}")

def main():
    dataset_dir = "./data/cholect50-challenge-val"
    video_id = random.choice(['VID68'])  # Replace with actual video IDs
    img_dir = os.path.join(dataset_dir, 'videos', video_id)
    label_file_path = os.path.join(dataset_dir, 'labels', f"{video_id}.json")

    # Load labels
    with open(label_file_path, 'r') as f:
        label_data = json.load(f)

    # Prepare labels in a more usable format
    labels = {}
    for frame in label_data['annotations']:
        labels[frame] = label_data['annotations'][frame]  # Each frame's labels

    # Draw bounding boxes on the image frames
    output_path = os.path.join("output", video_id)  # Specify output path
    draw_bounding_boxes(img_dir, labels, output_path)

if __name__ == "__main__":
    main()
