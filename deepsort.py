import json
import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
import threading
from shapely.geometry import Polygon
import bob.measure
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt

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
        cv2.waitKey(1000)
        if frame_index < len(frames):
            display_frame = cv2.resize(frames[frame_index], (frames[frame_index].shape[1] // 4, frames[frame_index].shape[0] // 4))
            cv2.imshow('Images with Bounding Box', display_frame)
            frame_index += 1
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Function to calculate IoU-based similarity scores for each probe
def calculate_scores(gt_boxes, predicted_boxes, iou_threshold=0.4):
    scores = []
    for gt_box in gt_boxes:
        iou_scores = [bb_intersection_over_union(gt_box, pred_box) for pred_box in predicted_boxes]
        
        # Separate positive (IoU > threshold) and negative (IoU <= threshold) scores
        positive_scores = [iou for iou in iou_scores if iou > iou_threshold]
        negative_scores = [iou for iou in iou_scores if iou <= iou_threshold]
        
        # Append the scores as a tuple (negatives, positives)
        scores.append((negative_scores, positive_scores))
    return scores

display_thread = threading.Thread(target=display_images)
display_thread.start()
VIDEO_NAME = "c6v5"

# Modify the track_instrument function to include CMC metric calculation
def track_instrument(cap, model, json_content):
    index = 0
    total_iou_array = []
    detections = []
    contours = None
    total_scores = []

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
            detections = []

            mask_gray = (mask_array > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 10), (0, 0, 255), 5)  # model gives red

                    detections.append([[x, y, w, h], 1.0, i])  # [x1, y1, w, h, confidence, class_id]
                    predicted_boxes.append([x, y, w, h])
                    i += 1

            frames.append(frame)
        else:
            # Update tracker with the detections
            tracks = deepsort.update_tracks(detections, frame=frame)

            detections = []

            for track in tracks:
                if track.is_confirmed():
                    bbox = track.to_tlbr()  # Get bounding box in (x1, y1, x2, y2) format
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)  # deepsort gives green

                    predicted_boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])  # tlwh format

                    if index % 10 != 1:
                        detections.append([[int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])], 1.0, i])

                    i += 1

            frames.append(frame)

        index += 1
        ground_truth_boxes = json_content.get(str(index), [])
        iou_array = []

        # Calculate scores for this frame
        frame_scores = calculate_scores(ground_truth_boxes, predicted_boxes)
        total_scores.extend(frame_scores)  # Store all scores across frames

        for detected_box in predicted_boxes:
            predicted_ious = []

            for gt_box in ground_truth_boxes:
                cv2.rectangle(frame, (gt_box[0], gt_box[1]), (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), (0, 0, 255), 5)
                iou = bb_intersection_over_union(detected_box, gt_box)
                if iou > 0.1 and iou < 1:
                    predicted_ious.append(iou)
            
            predicted_ious = sorted(predicted_ious, reverse=True)

            if (len(predicted_ious) > 0):
                iou_array.append(predicted_ious[0])
                total_iou_array.append(predicted_ious[0])

        print(f"F{index}: {np.average(iou_array): .2f}")

    print(f"Total IOU: {np.average(total_iou_array): .2f}")

    # Compute CMC curve
    print("Total Scores")
    print(total_scores)

    bob.measure.np.int = int
    cmc_values = bob.measure.cmc(total_scores)

    print("CMC Values")
    print(cmc_values)

    # Plot CMC curve
    ranks = np.arange(1, len(cmc_values) + 1)
    plt.plot(ranks, cmc_values)

    print("Ranks")
    print(ranks)

    plt.xlabel("Rank")
    plt.ylabel("Recognition Rate")
    plt.title("CMC Curve")
    plt.grid(True)
    plt.savefig("plot.png")

    cv2.destroyAllWindows()

# Example usage
model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

cap = cv2.VideoCapture(f"./data/videos/{VIDEO_NAME}.mp4")
json_content = []

with open(f'./data/videos/{VIDEO_NAME}_square.json', 'r') as json_file:
    json_content = json.load(json_file)

track_instrument(cap, model, json_content)
cap.release()

# Wait for display thread to finish
display_thread.join()
