import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
import threading

# Initialize Kalman Filter
kf = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)  # indicates that you want the x, y position of the object
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)

# Function to initialize Kalman Filter with the first detected position
def initialize_kalman_filter(x, y):
    kf.statePre = np.array([x, y, 0, 0], np.float32)
    kf.statePost = np.array([x, y, 0, 0], np.float32)
    kf.errorCovPre = np.eye(4, dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

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

frames = []

def display_video():
    frame_index = 0
    while True:
        cv2.waitKey(500)
        if frame_index < len(frames):
            display_frame = cv2.resize(frames[frame_index], (frames[frame_index].shape[1] // 4, frames[frame_index].shape[0] // 4))
            cv2.imshow('Video with Bounding Box', display_frame)
            frame_index += 1
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

display_thread = threading.Thread(target=display_video)
display_thread.start()

# Function to apply Kalman Filter and track the instrument
def track_instrument(video_path, model, model2=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    index = 0
    rectangles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if index % 10 == 0:
            rectangles = []
            # Update the Kalman Filter with the detected position
            input_image = preprocess_frame(frame)
            mask = model(input_image)
            mask_array = mask.data[0].cpu().numpy()[0] 
            y, x = np.where(mask_array > 0)
            
            # Predict the next position
            predicted_state = kf.predict()                
            measured_state = np.array([[np.mean(x)], [np.mean(y)]], np.float32)
            kf.correct(measured_state)

            # Find contours and draw bounding boxes
            mask_gray = (mask_array > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 3000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    rectangles.append([x,y,x+w,y+h])

            # Draw the predicted position
            predicted_x, predicted_y = predicted_state[0], predicted_state[1]
            print(predicted_x)
            cv2.circle(frame, (int(predicted_x), int(predicted_y)), 100, (0, 0, 255), 5)
            frames.append(frame)
        else:
            for rectangle in rectangles:
                cv2.rectangle(frame, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (0, 255, 0), 2)
            frames.append(frame)

        index += 1
        print("Finished processing frame", len(frames))
    
    cap.release()
    cv2.destroyAllWindows()

try:
    # Example usage
    model_path = 'data/models/unet11_binary_20/model_0.pt'
    model = get_model(model_path, model_type='UNet11', problem_type='binary')

    video_path = './data/videos/c6v4.mp4'
    track_instrument(video_path, model)

    # Wait for display thread to finish
    display_thread.join()
except Exception as Error:
    print(Error)
