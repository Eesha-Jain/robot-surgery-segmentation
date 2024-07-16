import cv2
import numpy as np
import torch
from generate_masks import get_model
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
import threading

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

def process_one_frame(frame, model):
    with torch.no_grad():
        transformed_image = img_transform(p=1)(image=frame)['image']
        input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)

    mask = model(input_image)
    mask_array = mask.data[0].cpu().numpy()[0]
    return mask_array > 0

def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01  # Decrease process noise
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1  # Decrease measurement noise
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1  # Set initial error covariance
    return kalman

# Get the model
model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

# Kalman filtering initialization
cap = cv2.VideoCapture("./data/videos/c6v5.mp4")
kalman = initialize_kalman_filter()

# Start image analysis
frames = []
mask = None
predicted_mask_position = None

def display_video():
    while True:
        if len(frames) > 0:
            display_frame = cv2.resize(frames[-1], (frames[-1].shape[1] // 4, frames[-1].shape[0] // 4))
            cv2.imshow('Video with Mask', display_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(100)

display_thread = threading.Thread(target=display_video)
display_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if mask is None:  # First frame: use Unet
        mask = process_one_frame(frame, model)
        mask_coords = np.column_stack(np.where(mask))
        initial_position = np.mean(mask_coords, axis=0).astype(np.float32)  # Calculate initial mask position
        kalman.statePre = np.array([initial_position[0], initial_position[1], 0, 0], np.float32)  # Initialize Kalman filter state
        kalman.statePost = kalman.statePre.copy()  # Ensure statePost is also initialized
        predicted_mask_position = initial_position  # Set initial predicted mask position
    else:  # Other frames: use Kalman
        measurement = np.mean(np.column_stack(np.where(mask)), axis=0).astype(np.float32)  # Calculate current mask position
        kalman.correct(measurement)  # Update Kalman filter with measurement
        predicted_mask_position = kalman.predict()[:2]  # Get predicted position from Kalman filter

    # Smooth the movement by limiting the shift amount
    max_shift = 5  # Maximum pixels to shift per frame
    dx = int(np.clip(predicted_mask_position[0] - mask_coords[0][0], -max_shift, max_shift))
    dy = int(np.clip(predicted_mask_position[1] - mask_coords[0][1], -max_shift, max_shift))
    mask = np.roll(mask, shift=dx, axis=0)  # Shift mask in x-axis
    mask = np.roll(mask, shift=dy, axis=1)  # Shift mask in y-axis

    # Append masked frame
    overlaid_frame = mask_overlay(frame, mask)
    frames.append(overlaid_frame)

    print("Finished processing frame", len(frames))

# Display the final video
cap.release()
cv2.destroyAllWindows()
