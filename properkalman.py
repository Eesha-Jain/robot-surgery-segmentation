import json
import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
import threading
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Initialize Kalman Filter
def initialize_kalman_filter(x=0, y=0):
    kf = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], np.float32)  # indicates that you want the x, y position of the object
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.statePre = np.array([x, y, 0, 0], np.float32)
    kf.statePost = np.array([x, y, 0, 0], np.float32)
    kf.errorCovPre = np.eye(4, dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

def mask_overlay(image, mask, color=(0, 255, 0)):
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)

def preprocess_frame(frame):
    transformed_image = img_transform(p=1)(image=frame)['image']
    input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)
    return input_image

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
centers_gt = []
centers_pred = []

def display_images():
    frame_index = 0
    while True:
        cv2.waitKey(500)
        if frame_index < len(frames):
            display_frame = cv2.resize(frames[frame_index], (frames[frame_index].shape[1] // 4, frames[frame_index].shape[0] // 4))
            cv2.imshow('Images with Bounding Box', display_frame)
            frame_index += 1
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

display_thread = threading.Thread(target=display_images)
display_thread.start()
VIDEO_NAME = "c6v5"
kfs = [initialize_kalman_filter(), initialize_kalman_filter(), initialize_kalman_filter(), initialize_kalman_filter(), initialize_kalman_filter()]

def track_instrument(cap, model, json_content):
    index = 0
    rectangles = []
    total_iou_array = []
    contours = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        predicted_boxes = []
        if index % 10 == 0:
            rectangles = []
            input_image = preprocess_frame(frame)
            mask = model(input_image)
            mask_array = mask.data[0].cpu().numpy()[0]
            y, x = np.where(mask_array > 0)

            mask_gray = (mask_array > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            i = 0
            for contour in contours:
                if cv2.contourArea(contour) > 15000 and len(kfs) > i:
                    x, y, w, h = cv2.boundingRect(contour)
                    measurement = np.array([[np.float32(x)], [np.float32(y)]])
                    kfs[i].correct(measurement)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    rectangles.append([x, y, w, h])
                    predicted_boxes.append([x, y, w, h])
                    i += 1

            frames.append(frame)
        else:
            for i in range(len(rectangles)):
                if len(kfs) > i:
                    x, y, w, h = rectangles[i][0], rectangles[i][1], rectangles[i][2], rectangles[i][3]
                    kfs[i].correct(np.array([[np.float32(x+w/2)], [np.float32(y+h/2)]]))

                    prediction = kfs[i].predict()
                    pred_x, pred_y = prediction[0]-w/2, prediction[1]-h/2
                    cv2.rectangle(frame, (int(pred_x), int(pred_y)), (int(pred_x + w), int(pred_y + h)), (0, 255, 0), 5)
                    predicted_boxes.append([pred_x, pred_y, w, h])

            frames.append(frame)

        index += 1
        ground_truth_boxes = json_content[str(index)]
        iou_array = []
        for gt_box in ground_truth_boxes:
            cv2.rectangle(frame, (gt_box[0], gt_box[1]), (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), (0, 0, 255), 5)
            for detected_box in predicted_boxes:
                iou = bb_intersection_over_union(detected_box, gt_box)
                if iou > 0.1 and iou < 1:
                    iou_array.append(iou)
                    total_iou_array.append(iou)
        
        print(f"F{index}: {np.average(iou_array): .2f}")

    print(f"Total IOU: {np.average(total_iou_array): .2f}")
    cv2.destroyAllWindows()

# Example usage
model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

cap = cv2.VideoCapture(f"./data/videos/{VIDEO_NAME}.mp4")
json_content = []

with open(f'./data/videos/{VIDEO_NAME}.json', 'r') as json_file:
    json_content = json.load(json_file)

track_instrument(cap, model, json_content)
cap.release()

# Wait for display thread to finish
display_thread.join()
