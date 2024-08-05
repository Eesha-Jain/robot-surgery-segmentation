import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
import threading
import os
from pathlib import Path

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)  # indicates that you want the x, y position of the object
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
      
    def set_bounding_box(self, x, y, w, h):
        self.kf.statePre = np.array([x, y, 0, 0], np.float32)
        self.kf.statePost = np.array([x, y, 0, 0], np.float32)
        self.kf.errorCovPre = np.eye(4, dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.x = x
        self.y = y
        self.w = w 
        self.h = h
        self.correct(x, y)

    def correct(self, x, y):
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))

    def predict(self):
        prediction = self.kf.predict()
        pred_x, pred_y = prediction[0], prediction[1]
        self.x = pred_x
        self.y = pred_y
        return (pred_x, pred_y)

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
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

def extract_bounding_boxes(mask, frame):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    unique_shades = np.unique(mask_gray)
    boxes = []
    
    for shade in unique_shades:
        if shade == 0: #skip background
            continue
        
        binary_mask = np.uint8(mask_gray == shade) * 255
        x, y, w, h = cv2.boundingRect(binary_mask)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        boxes.append([x, y, w, h])
    
    return boxes

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)
    
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3] - boxB[1]
    
    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

frames = []

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

def track_instrument(dir, model):
    frame_files = sorted(Path(dir + "images").glob('*.jpg'))
    ground_truth_files = sorted(Path(dir + "instruments_masks").glob('*.png'))
    index = 0
    rectangles = []
    contours = None

    for i in range(len(frame_files)):
        frame = cv2.imread(str(frame_files[i]))
        ground_truth = cv2.imread(str(ground_truth_files[i]))
        if frame is None:
            print(f"Failed to read {frame_files[i]}")
            continue

        if index % 10 == 0:
            rectangles = []
            # Update the Kalman Filter with the detected position
            input_image = preprocess_frame(frame)
            mask = model(input_image)
            mask_array = mask.data[0].cpu().numpy()[0]
            y, x = np.where(mask_array > 0)

            # Find contours and draw bounding boxes
            mask_gray = (mask_array > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 6000:
                    x, y, w, h = cv2.boundingRect(contour)
                    kf = KalmanFilter()
                    kf.set_bounding_box(x, y, w, h)
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    rectangles.append(kf)

            frames.append(frame)
        else:
            for i in range(len(contours)):
                if len(rectangles) > i:
                    kf = rectangles[i]
                    x, y, w, h = kf.x, kf.y, kf.w, kf.h
                    kf.correct(x, y)

                    # Predict position of instruments
                    pred_x, pred_y = kf.predict()
                    cv2.rectangle(frame, (int(pred_x), int(pred_y)), (int(pred_x + w), int(pred_y + h)), (0, 255, 0), 5)

            frames.append(frame)
        
        #calculate iou in the frame
        ground_truth_boxes = extract_bounding_boxes(ground_truth, frame)
        iou_array = []
        for kf in rectangles:
            for gt_box in ground_truth_boxes:
                detected_box = [kf.x, kf.y, kf.w, kf.h]
                iou = bb_intersection_over_union(detected_box, gt_box)
                if iou > 0.1 and iou < 1:
                    iou_array.append(iou)
                    # print(f"IoU: {iou:.2f}, Detected box: {detected_box}, Ground truth box: {gt_box}")

        print(f"AVERAGE IOU FOR FRAME {len(frames)}: {np.average(iou_array): .2f}")
        index += 1

    cv2.destroyAllWindows()

# Example usage
model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

dir = './data/cropped_train/instrument_dataset_1/'
track_instrument(dir, model)

# Wait for display thread to finish
display_thread.join()