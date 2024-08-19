from pylab import *
import cv2
from dataset import load_image
import torch
import numpy as np
import albumentations
from utils import cuda
from generate_masks import get_model
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
import threading
import time
import json

rcParams['figure.figsize'] = 10, 10

### KALMAN FILTER INITIALIZATION
kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)  # indicates that you want the x, y position of the object
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)  # says dx influences x, dy influences y, dx & dy are not influenced by anything

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)

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

def image_touchup(frame):
    with torch.no_grad():
        transformed_image = img_transform(p=1)(image=frame)['image']
        input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)
    
    return input_image

model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

VIDEO_NAME = "c7v11"
cap = cv2.VideoCapture(f"./data/videos/{VIDEO_NAME}.mp4")

frames = []
rectangles_per_frame = {}
index = 0

def display_video():
    frame_index = 0
    while True:
        cv2.waitKey(100)
        if frame_index < len(frames):
            display_frame = cv2.resize(frames[frame_index], (frames[frame_index].shape[1] // 4, frames[frame_index].shape[0] // 4))
            cv2.imshow('Video with Bounding Box', display_frame)
            frame_index += 1
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# display_thread = threading.Thread(target=display_video)
# display_thread.start()

def runProcessing():
    global rectangles_per_frame
    global index 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame with the model
        mask = model(image_touchup(frame))
        mask_array = mask.data[0].cpu().numpy()[0]
        mask_gray = (mask_array > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15000:
                x2, y2, width, height = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x2, y2), (x2 + width, y2 + height), (0, 255, 0), 5)
                rectangles.append([x2, y2, width, height])
        
        # Save the rectangles for this frame
        rectangles_per_frame[index + 1] = rectangles

        frames.append(frame)
        print("Finished processing frame", index + 1)
        index += 1

# Start processing
runProcessing()
cap.release()
cv2.destroyAllWindows()

# Save results to JSON
with open(f'./data/videos/{VIDEO_NAME}.json', 'w') as json_file:
    json.dump(rectangles_per_frame, json_file, indent=4)
print("Output saved to output.json")