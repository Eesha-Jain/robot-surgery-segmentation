import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
import threading
import os
from pathlib import Path
import json
import random

# Initialize Kalman Filter
kf = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)

def crop_to_square(image):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    
    # Calculate the center of the image
    center_x, center_y = w // 2, h // 2
    half_dim = min_dim // 2
    
    # Define the square coordinates
    x1 = center_x - half_dim
    y1 = center_y - half_dim
    x2 = center_x + half_dim
    y2 = center_y + half_dim
    
    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Crop the image
    cropped_image = image[y1:y2, x1:x2]
    
    return cropped_image, (x1, y1)

def preprocess_frame(frame, target_size=(512, 512)):
    # Crop the frame to a square
    cropped_frame, (offset_x, offset_y) = crop_to_square(frame)
    
    # Resize the cropped frame to the target size
    frame_resized = cv2.resize(cropped_frame, target_size)
    transformed_image = img_transform(p=1)(image=frame_resized)['image']
    input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)
    
    return input_image, (offset_x, offset_y)

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = (xB - xA) * (yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

frames = []

def display_images():
    frame_index = 0
    while True:
        if frame_index < len(frames):
            # Show the full-sized frame
            cv2.imshow('Images with Bounding Box', frames[frame_index])
            frame_index += 1
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.waitKey(1000)  # Display each frame for 1 second

    cv2.destroyAllWindows()

display_thread = threading.Thread(target=display_images)
display_thread.start()

def track_instrument(dir, model, labels, target_size=(512, 512)):
    img_dir = os.path.join(dir, 'videos', 'VID68')  # Change to your specific video ID
    frame_files = sorted(Path(img_dir).glob('*.png'))
    index = 0
    rectangles = []
    total_iou_array = []

    for i in range(len(frame_files)):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None:
            print(f"Failed to read {frame_files[i]}")
            continue

        # Process every frame
        input_image, (offset_x, offset_y) = preprocess_frame(frame, target_size=target_size)
        mask = model(input_image)
        mask_array = mask.data[0].cpu().numpy()[0]
        mask_gray = (mask_array > 0).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []

        # Draw predicted boxes
        for contour in contours:
            if cv2.contourArea(contour) > 6000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                # Adjust for the offset from cropping
                measurement = np.array([[np.float32(x + offset_x)], [np.float32(y + offset_y)]])
                kf.correct(measurement)
                cv2.rectangle(frame, (x + offset_x, y + offset_y), (x + offset_x + w, y + offset_y + h), (0, 255, 255), 5)
                rectangles.append([x + offset_x, y + offset_y, w, h])

        # Extract ground truth boxes from labels
        frame_index = str(index)
        ground_truth_boxes = []
        if frame_index in labels:
            for label in labels[frame_index]:
                gt_x1 = label[3] * frame.shape[1]  # Scale to original image dimensions
                gt_y1 = label[4] * frame.shape[0]
                gt_bw = label[5] * frame.shape[1]
                gt_bh = label[6] * frame.shape[0]

                # Adjust ground truth bounding boxes for the crop
                gt_x1 = int(gt_x1 - offset_x)
                gt_y1 = int(gt_y1 - offset_y)

                # Ensure ground truth boxes are within frame bounds
                gt_x1 = max(0, gt_x1)
                gt_y1 = max(0, gt_y1)
                ground_truth_boxes.append([gt_x1, gt_y1, int(gt_bw), int(gt_bh)])

        # Draw the ground truth bounding boxes
        for gt_box in ground_truth_boxes:
            cv2.rectangle(frame, (gt_box[0], gt_box[1]), (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), (0, 0, 255), 2)

        frames.append(frame)
        index += 1

    print(f"Total IOU: {np.average(total_iou_array): .2f}")
    cv2.destroyAllWindows()

# Load the model
model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

# Load labels from Cholec50 dataset
dataset_dir = "./data/cholect50-challenge-val"
video_id = 'VID68'  # Specify your video ID
label_file_path = os.path.join(dataset_dir, 'labels', f"{video_id}.json")

# Load labels
with open(label_file_path, 'r') as f:
    label_data = json.load(f)

# Prepare labels in a more usable format
labels = {}
for frame in label_data['annotations']:
    labels[frame] = label_data['annotations'][frame]  # Each frame's labels

# Run tracking
track_instrument(dataset_dir, model, labels)

# Wait for display thread to finish
display_thread.join()
