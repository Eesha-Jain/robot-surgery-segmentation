from pylab import *
import cv2

rcParams['figure.figsize'] = 10, 10

from dataset import load_image
import torch
import albumentations
from utils import cuda
from generate_masks import get_model
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip


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

def process_one_frame(frame, model):
    with torch.no_grad():
        transformed_image = img_transform(p=1)(image=frame)['image']
        input_image = torch.unsqueeze(ToTensor()(transformed_image), dim=0)

    mask = model(input_image)
    mask_array = mask.data[0].cpu().numpy()[0]
    return mask_overlay(frame, (mask_array > 0).astype(np.uint8))

video_path = "./data/videos/c4v4.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = process_one_frame(frame, model)

    plt.imshow(frame)
    plt.show()

    frames.append(frame)

cap.release()
clip = ImageSequenceClip(frames, fps=30)
clip.ipython_display()