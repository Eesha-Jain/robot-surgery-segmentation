from pylab import *
import cv2
from dataset import load_image
import torch
import albumentations
from utils import cuda
from generate_masks import get_model
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
import threading
import time

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

def adjust_bounding_box(contour, buffer, frame_width, frame_height):
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([
        [[max(x - buffer, 0), max(y - buffer, 0)]],
        [[min(x + w + buffer, frame_width), max(y - buffer, 0)]],
        [[min(x + w + buffer, frame_width), min(y + h + buffer, frame_height)]],
        [[max(x - buffer, 0), min(y + h + buffer, frame_height)]]
    ], dtype=np.int32)

def image_touchup(frame):
    with torch.no_grad():
        transformed_image = img_transform(p=1)(image=frame)['image']
        input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)
        input_image = input_image.to("cuda")
    
    return input_image

model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

cap = cv2.VideoCapture("./data/videos/c6v4.mp4")

frames = []
mask = None
mask_array = None
rectangles = []
buffer = 20

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

def runProcessing():
    ret, frame = cap.read()
    if not ret:
        exit

    index = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    mask = model(image_touchup(frame))
    mask_array = mask.data[0].cpu().numpy()[0]
    overlay = mask_overlay(frame, (mask_array > 0).astype(np.uint8))
    frames.append(np.hstack((overlay, overlay)))
    mask_gray = (mask_array > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rectangles = []
        contours_copy = []

        if index == 0:
            for cnt in contours:
                if cv2.contourArea(cnt) > 3000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    rect_contour = adjust_bounding_box(cnt, 0, frame_width, frame_height)
                    contours_copy.append(rect_contour)
                    rectangles.append([x,y,w,h])
                    
            frames.append(np.hstack((frame, overlay)))
            contours = contours_copy
        elif index % 10 == 0:
            for cnt in contours:
                if cv2.contourArea(cnt) > 3000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = frame[y:y+h, x:x+w]

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    contours2, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt2 in contours2:
                        if cv2.contourArea(cnt2) > 3000:
                            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
                            x2 += x
                            y2 += y
                            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                            rectangles.append([x2, y2, w2, h2])
                            rect_contour = adjust_bounding_box(cnt2, buffer, frame_width, frame_height)
                            contours_copy.append(rect_contour)
            
            frames.append(np.hstack((frame, overlay)))
            contours = contours_copy
        else:
            for rectangle in rectangles:
                x, y, w, h = rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            frames.append(np.hstack((frame, overlay)))

        print("Finished processing frame", len(frames))
        index += 1

# Display the final video
runProcessing()
cap.release()
cv2.destroyAllWindows()