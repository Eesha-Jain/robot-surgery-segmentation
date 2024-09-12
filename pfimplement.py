import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize
from torchvision.transforms import ToTensor
from generate_masks import get_model
from numpy.random import uniform, randn
import threading
from shapely.geometry import Polygon
from numpy.random import randn
import json
from filterpy.monte_carlo import systematic_resample

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def predict(particles, u, std, dt=1.):
    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

    return particles

def update(particles, weights, z, R):
    weights += 1.e-300
    weights /= sum(weights)
    return weights

def estimate(particles, weights):
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def resample_from_index(particles, weights, indexes):
    weights = np.array(weights)
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

def neff(weights):
    return 1. / np.sum(np.square(weights))

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

def draw_rotated_box(image, box, color=(0, 255, 0)):
    center, (width, height), angle = box
    box_points = cv2.boxPoints(box)
    box_points = np.intp(box_points)
    cv2.drawContours(image, [box_points], 0, color, 5)
    return image

def box_to_polygon(box):
    center, (width, height), angle = box
    rect = cv2.boxPoints(box)
    rect = np.intp(rect)
    return Polygon(rect)

def bb_intersection_over_union(boxA, boxB):
    # Convert the boxes to polygons
    polyA = box_to_polygon(boxA)
    polyB = box_to_polygon(boxB)
    
    # Compute the intersection and union of the polygons
    intersection_poly = polyA.intersection(polyB)
    union_poly = polyA.union(polyB)
    
    # Calculate areas
    inter_area = intersection_poly.area
    union_area = union_poly.area
    
    if union_area == 0:
        return 0
    
    iou = inter_area / union_area
    return iou

frames = []
VIDEO_NAME = "c7v11"

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

def track_instrument(cap, model, json_content, N=1000):
    rectangles = []
    total_iou_array = []
    
    particles = []
    weights = []
    index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if index % 10 == 0:
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
                if cv2.contourArea(contour) > 15000:
                    rect = cv2.minAreaRect(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    particle = create_uniform_particles((x, x + w), (y, y + h), (0, 6.28), N)
                    particles.append(particle)
                    
                    weight = np.array([.25]*1000)
                    weights.append(weight)

                    estimate(particle, weight)

                    for p in particle:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 0), 5)
                    
                    rectangles.append(rect)
                    frame = draw_rotated_box(frame, rect, color=(0, 255, 0))
        else:
            rectangles_2 = []
            for j in range(len(rectangles)):
                rect = rectangles[j]
                particle = particles[j]
                weight = weights[j]
                
                particles[j] = predict(particle, u=(0.00, 1.414), std=(.2, .05))
                measurements = np.array([rect[0][0], rect[0][1]])
                weights[j] = update(particle, weight, z=measurements, R=0.1)
                
                if neff(weights) < N/2:
                    indexes = systematic_resample(weight)
                    resample_from_index(particle, weights, indexes)
                    assert np.allclose(weight, 1/N)
                mu, var = estimate(particle, weight)

                center = (mu[0], mu[1])
                width = rect[1][0]
                height = rect[1][1]
                angle = rect[2]

                rectangles_2.append((center, (width, height), angle))
                frame = draw_rotated_box(frame, (center, (width, height), angle), color=(0, 255, 0))
                
                for p in particle:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 0), 5)
            
            rectangles = rectangles_2

        index += 1
        ground_truth_boxes = json_content.get(str(index), [])
        iou_array = []
        for gt_box in ground_truth_boxes:
            gt_center = (gt_box[0], gt_box[1])
            gt_width = gt_box[2]
            gt_height = gt_box[3]
            gt_angle = gt_box[4]
            gt_rect = (gt_center, (gt_width, gt_height), gt_angle)
            cv2.drawContours(frame, [cv2.boxPoints(gt_rect).astype(np.intp)], 0, (0, 0, 255), 5)
            for detected_box in rectangles:
                iou = bb_intersection_over_union(detected_box, gt_rect)
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

cap = cv2.VideoCapture(f"./data/videos/{VIDEO_NAME}.mp4")
json_content = []

with open(f'./data/videos/{VIDEO_NAME}_tilt.json', 'r') as json_file:
    json_content = json.load(json_file)

track_instrument(cap, model, json_content)
cap.release()

# Wait for display thread to finish
display_thread.join()
