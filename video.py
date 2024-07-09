from pylab import *
import cv2
from dataset import load_image
import torch
import albumentations
from utils import cuda
from generate_masks import get_model
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

rcParams['figure.figsize'] = 10, 10

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

def kalman_filtering(measurement, kalman_filter):
    kalman_filter.correct(measurement) #update mask with new value
    predicted_state = kalman_filter.predict() #predict where the mask will be next
    return predicted_state

#get the model
model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

#kalman filtering initialization
cap = cv2.VideoCapture("./data/videos/c4v4.mp4")
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

#start image analysis
frames = []
mask = None
predicted_mask_position = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mask is None: #first frame: use Unet
        mask = process_one_frame(frame, model)
        mask_coords = np.column_stack(np.where(mask))
        initial_position = np.mean(mask_coords, axis=0).astype(np.float32) #Calculate initial mask position
        kalman.statePre = np.array([initial_position[0], initial_position[1], 0, 0], np.float32)  # Initialize Kalman filter state
        predicted_mask_position = initial_position  # Set initial predicted mask position
    else: #other frame: use Kalman
        measurement = np.mean(np.column_stack(np.where(mask)), axis=0).astype(np.float32)  # Calculate current mask position
        predicted_mask_position = kalman_filtering(measurement, kalman)[:2]  # Get predicted position from Kalman filter

    #shift mask to the predicted position (as calculated above)
    dx = int(predicted_mask_position[0] - mask_coords[0][0])
    dy = int(predicted_mask_position[1] - mask_coords[0][1])
    mask = np.roll(mask, shift=dx, axis=0)  # Shift mask in x-axis
    mask = np.roll(mask, shift=dy, axis=1)  # Shift mask in y-axis

    #append masked frame
    overlaid_frame = mask_overlay(frame, mask)
    frames.append(overlaid_frame)

    print("Finished processing frame {}", len(frames))

#display the final video
cap.release()
clip = ImageSequenceClip(frames, fps=30)
clip.ipython_display()