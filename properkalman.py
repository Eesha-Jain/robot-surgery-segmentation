import json
import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
import threading
from shapely.geometry import Polygon
import time

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

def draw_rotated_box(image, box, color=(0, 255, 0)):
    center, (width, height), angle = box
    box_points = cv2.boxPoints(box)
    box_points = np.intp(box_points)
    cv2.drawContours(image, [box_points], 0, color, 5)
    return image

def box_to_polygon(box):
    center, (width, height), angle = box
    rect = cv2.boxPoints(box)
    rect = np.intp(rect)
    return Polygon(rect)

def bb_intersection_over_union(boxA, boxB):
    # Convert the boxes to polygons
    polyA = box_to_polygon(boxA)
    polyB = box_to_polygon(boxB)
    
    # Compute the intersection and union of the polygons
    intersection_poly = polyA.intersection(polyB)
    union_poly = polyA.union(polyB)
    
    # Calculate areas
    inter_area = intersection_poly.area
    union_area = union_poly.area
    
    if union_area == 0:
        return 0
    
    iou = inter_area / union_area
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
            t0 = time.time()
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
                    rect = cv2.minAreaRect(contour)
                    frame = draw_rotated_box(frame, rect, color=(0, 255, 0))
                    kfs[i].correct(np.array([[np.float32(rect[0][0] + rect[1][0]/2)], [np.float32(rect[1][0] + rect[1][1]/2)]]))
                    
                    rectangles.append(rect)
                    predicted_boxes.append(rect)
                    i += 1

            frames.append(frame)
            t1 = time.time()

            print(t1-t0)
        else:
            t0 = time.time()
            for i in range(len(rectangles)):
                if len(kfs) > i:
                    rect = rectangles[i]
                    center, (width, height), angle = rect
                    kfs[i].correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))

                    prediction = kfs[i].predict()
                    pred_center = (prediction[0], prediction[1])
                    pred_rect = (pred_center, (width, height), angle)
                    
                    frame = draw_rotated_box(frame, pred_rect, color=(0, 255, 0))
                    predicted_boxes.append(pred_rect)

            frames.append(frame)
            t1 = time.time()
            print(t1 - t0)

        index += 1
        ground_truth_boxes = json_content.get(str(index), [])
        iou_array = []
        for gt_box in ground_truth_boxes:
            gt_center = (gt_box[0], gt_box[1])
            gt_width = gt_box[2]
            gt_height = gt_box[3]
            gt_angle = gt_box[4]
            gt_rect = (gt_center, (gt_width, gt_height), gt_angle)
            cv2.drawContours(frame, [cv2.boxPoints(gt_rect).astype(np.intp)], 0, (0, 0, 255), 5)
            for detected_box in predicted_boxes:
                iou = bb_intersection_over_union(detected_box, gt_rect)
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

with open(f'./data/videos/{VIDEO_NAME}_tilt.json', 'r') as json_file:
    json_content = json.load(json_file)

track_instrument(cap, model, json_content)
cap.release()

# Wait for display thread to finish
display_thread.join()