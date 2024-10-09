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
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSORT Tracker
deepsort = DeepSort(max_age=30, n_init=1, nms_max_overlap=1.0, max_cosine_distance=0.7)

def mask_overlay(image, mask, color=(0, 255, 0)):
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def img_transform(p=1):
    return Compose([Normalize(p=1)], p=p)

def preprocess_frame(frame):
    transformed_image = img_transform(p=1)(image=frame)['image']
    input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)
    return input_image

def get_boxA_corners(boxA):
    # boxA is [x1, y1, x2, y2]
    x1, y1, x2, y2 = boxA
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Return corner points for axis-aligned boxA
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), width, height, (center_x, center_y)

def get_boxB_corners(center, width, height, angle):
    # boxB is [centerX, centerY, width, height, angle]
    cx, cy = center
    rect = ((cx, cy), (width, height), angle)
    
    # Get corner points using OpenCV's boxPoints
    corners = cv2.boxPoints(rect)
    return np.array(corners)

def bb_intersection_over_union(boxA, boxB):
    # Extract data for boxA (axis-aligned)
    boxA_corners, widthA, heightA, centerA = get_boxA_corners(boxA)
    
    # Extract data for boxB (rotated box)
    centerBX, centerBY, widthB, heightB, angleB = boxB
    boxB_corners = get_boxB_corners((centerBX, centerBY), widthB, heightB, angleB)
    
    # Convert to polygon shapes
    polyA = np.array([boxA_corners], dtype=np.int32)
    polyB = np.array([boxB_corners], dtype=np.int32)
    
    # Compute intersection area using cv2.intersectConvexConvex (if OpenCV version supports it)
    int_area, _ = cv2.intersectConvexConvex(polyA.astype(np.float32), polyB.astype(np.float32))
    
    # Compute the area of both boxes
    areaA = widthA * heightA
    areaB = widthB * heightB
    
    # Compute union area
    union_area = areaA + areaB - int_area
    
    # IoU is intersection over union
    iou = int_area / union_area
    return iou

frames = []
centers_gt = []
centers_pred = []

def display_images():
    frame_index = 0
    while True:
        cv2.waitKey(1000)
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

def track_instrument(cap, model, json_content):
    index = 0
    total_iou_array = []
    detections = []
    contours = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        predicted_boxes = []
        i = 0

        if index % 10 == 0:
            input_image = preprocess_frame(frame)
            mask = model(input_image)
            mask_array = mask.data[0].cpu().numpy()[0]
            y, x = np.where(mask_array > 0)

            mask_gray = (mask_array > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 10), (0, 0, 255), 5) #model gives red

                    detections.append([[x, y, w, h], 1.0, i])  # [x1, y1, w, h, confidence, class_id]
                    predicted_boxes.append([x, y, w, h])
                    i += 1

            frames.append(frame)
        else:
            # Update tracker with the detections
            tracks = deepsort.update_tracks(detections, frame=frame)
            
            if index % 10 != 1:
                detections = []
            
            for track in tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    bbox = track.to_tlbr()  # Get bounding box in (x1, y1, x2, y2) format
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3) #deepsort gives green
                    
                    predicted_boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])  # tlwh format
 
                    if index % 10 != 1:
                        detections.append([[int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])], 1.0, i])

                    i += 1

            frames.append(frame)

        index += 1
        ground_truth_boxes = json_content.get(str(index), [])
        iou_array = []
        for gt_box in ground_truth_boxes:
            gt_center = (gt_box[0], gt_box[1])
            gt_width = gt_box[2]
            gt_height = gt_box[3]
            gt_angle = gt_box[4]
            gt_rect = (gt_center, (gt_width, gt_height), gt_angle)
            cv2.drawContours(frame, [cv2.boxPoints(gt_rect).astype(np.intp)], 0, (255, 0, 0), 3) #ground truth is blue

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

with open(f'./data/videos/{VIDEO_NAME}_tilt.json', 'r') as json_file:
    json_content = json.load(json_file)

track_instrument(cap, model, json_content)
cap.release()

# Wait for display thread to finish
display_thread.join()
