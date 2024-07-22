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
    while True:
        if len(frames) > 0:
            display_frame = cv2.resize(frames[-1], (frames[-1].shape[1] // 4, frames[-1].shape[0] // 4))
            cv2.imshow('Video with Bounding Box', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(100)

display_thread = threading.Thread(target=display_video)
display_thread.start()

# Function to apply Kalman Filter and track the instrument
def track_instrument(video_path, model, model2=None):
    # cap = cv2.VideoCapture(video_path)
    # ret, frame = cap.read()
    # if not ret:
    #     print("Failed to read video")
    #     return
    
    # # Detect instrument in the first frame
    # input_image = preprocess_frame(frame)
    # mask = model(input_image)
    # mask_array = mask.data[0].cpu().numpy()[0]
    # y, x = np.where(mask_array > 0)
    # if len(x) == 0 or len(y) == 0:
    #     print("Instrument not detected in the first frame")
    #     return

    # Initialize Kalman Filter with the first detected position
    kf = initialize_kalman_filter(np.mean(x), np.mean(y))
    index = 0
    rectangles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if index % 10 == 0:
            # Predict the next position
            predicted_state = kf.predict()
            rectangles = []

            # Update the Kalman Filter with the detected position
            input_image = preprocess_frame(frame)
            mask = model(input_image)
            mask_array = mask.data[0].cpu().numpy()[0]
            y, x = np.where(mask_array > 0)
            if len(x) > 0 and len(y) > 0:
                measured_state = np.array([[np.mean(x)], [np.mean(y)]], np.float32)
                kf.correct(measured_state)

                # Find contours and draw bounding boxes
                contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x_min, y_min, width, height = cv2.boundingRect(contour)
                    x_max, y_max = x_min + width, y_min + height
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    rectangles.append([x_min, y_min, x_max, y_max])

            # Draw the predicted position
            predicted_x, predicted_y = predicted_state[0], predicted_state[1]
            cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (0, 0, 255), -1)
        else:
            for rectangle in rectangles:
                cv2.rectangle(frame, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (0, 255, 0), 2)

        # Add the frame to the frames list
        frames.append(frame)
        print("Finished processing frame")
        index += 1
    
    cap.release()
    cv2.destroyAllWindows()

try:
    # Example usage
    model_path = 'data/models/unet11_binary_20/model_0.pt'
    model = get_model(model_path, model_type='UNet11', problem_type='binary')
    #model2 = get_model(model_path, model_type='UNet11Simple', problem_type='binary')

    video_path = './data/videos/c6v4.mp4'
    track_instrument(video_path, model)

    # Wait for display thread to finish
    display_thread.join()
except Exception as Error:
    print(Error)