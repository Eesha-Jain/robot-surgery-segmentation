import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
from numpy.random import uniform, randn
from pathlib import Path
import threading
from scipy.stats import norm
from filterpy.monte_carlo import systematic_resample

def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles

def predict(particles, u, std):
    N = len(particles)
    particles[:, 0] += (u[0] + randn(N) * std[0])
    particles[:, 1] += (u[1] + randn(N) * std[1])

def update(particles, weights, z, R):
    distances = np.linalg.norm(particles - z, axis=1)
    weights *= norm.pdf(distances, 0, R)
    weights += 1.e-300  # Avoid zero weights
    weights /= np.sum(weights)  # Normalize

def estimate(particles, weights):
    mean = np.average(particles, weights=weights, axis=0)
    return mean

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = 1.0 / len(weights)

def neff(weights):
    return 1.0 / np.sum(np.square(weights))

def mask_overlay(image, mask, color=(0, 255, 0)):
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
    
    interArea = (xB - xA) * (yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
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

def track_instrument(dir, model, N=1000):
    frame_files = sorted(Path(dir + "images").glob('*.jpg'))
    ground_truth_files = sorted(Path(dir + "instruments_masks").glob('*.png'))
    rectangles = []
    total_iou_array = []
    
    particles = []
    weights = []
    
    for i in range(len(frame_files)):
        frame = cv2.imread(str(frame_files[i]))
        ground_truth = cv2.imread(str(ground_truth_files[i]))
        if frame is None:
            print(f"Failed to read {frame_files[i]}")
            continue
        
        if i % 10 == 0:
            rectangles = []
            if particles:
                particles = []
                weights = []
            
            input_image = preprocess_frame(frame)
            mask = model(input_image)
            mask_array = mask.data[0].cpu().numpy()[0]
            y, x = np.where(mask_array > 0)

            mask_gray = (mask_array > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 6000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    particle = create_uniform_particles((x, x + w), (y, y + h), N)
                    particles.append(particle)
                    weights.append(np.ones(N) / N)

                    for p in particle:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 0), 5)
                    
                    rectangles.append([x, y, w, h])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        else:
            rectangles_2 = []
            for j in range(len(rectangles)):
                x, y, w, h = rectangles[j]
                particle = particles[j]
                weight = weights[j]
                
                predict(particle, u=(5, 5), std=(1, 1))
                
                measurements = np.array([x + w / 2, y + h / 2])
                update(particle, weight, z=measurements, R=1000)
                
                if neff(weight) < N / 2:
                    indexes = systematic_resample(weight)
                    resample_from_index(particle, weight, indexes)
                
                mu = estimate(particle, weight)
                rectangles_2.append([mu[0] - w / 2, mu[1] - h / 2, w, h])
                cv2.rectangle(frame, (int(mu[0] - w / 2), int(mu[1] - h / 2)),
                                (int(mu[0] + w / 2), int(mu[1] + h / 2)), (0, 255, 0), 5)
                
                for p in particle:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 0), 5)
            
            rectangles = rectangles_2

        ground_truth_boxes = extract_bounding_boxes(ground_truth, frame)
        iou_array = []
        for detected_box in rectangles:
            for gt_box in ground_truth_boxes:
                iou = bb_intersection_over_union(detected_box, gt_box)
                if iou > 0.1 and iou < 1:
                    iou_array.append(iou)
                    total_iou_array.append(iou)
        
        frames.append(frame)
        print(f"F{len(frames)}: {np.average(iou_array): .2f}")
    
    print(f"Total IOU: {np.average(total_iou_array): .2f}")
    cv2.destroyAllWindows()

# Example usage
model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

dir = './data/cropped_train/instrument_dataset_1/'
track_instrument(dir, model)
