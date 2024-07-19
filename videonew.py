from pylab import *
import cv2
from dataset import load_image
import torch
import albumentations
from utils import cuda
from generate_masks import get_model
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor

rcParams['figure.figsize'] = 10, 10

### KALMAN FILTER INITIALIZATION
kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32) #indicates that you want the x,y position of the object
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32) #says dx influences x, dy influences y, dx & dy r not influenced by anything

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

model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

cap = cv2.VideoCapture("./data/videos/c6v4.mp4")

frames = []
mask = None
mask_array = None
rectangles = []
index = 0

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

    with torch.no_grad():
        transformed_image = img_transform(p=1)(image=frame)['image']
        input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)

    if index % 10 == 0: 
        if mask == None:
            mask = model(input_image)
            mask_array = mask.data[0].cpu().numpy()[0]
            overlay = mask_overlay(frame, (mask_array > 0).astype(np.uint8))
            frames.append(np.hstack((overlay, overlay)))
        else:
            mask_gray = (mask_array > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 3000:
                    ### KALMAN FILTER PREDICTION
                    prediction = kalman.predict()
                    x, y, w, h = cv2.boundingRect(cnt)

                    ### KALMAN FILTER CORRECTION
                    measurement = np.array([[x + w / 2], [y + h / 2]], np.float32)
                    if measurement[0][0] != 0 and measurement[1][0] != 0:
                        kalman.correct(measurement)
                        last_measurement = measurement
                    else:
                        measurement = last_measurement

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    rectangles.append([x, y, w, h])

            frames.append(np.hstack((frame, overlay)))
    else:
        for rectangle in rectangles:
            x, y, w, h = rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

    print("Finished processing frame", len(frames))
    index += 1

# Display the final video
cap.release()
cv2.destroyAllWindows()