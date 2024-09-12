import cv2
import json
import numpy as np

# Constants
VIDEO_NAME = 'c6v5'
INPUT_JSON_PATH = f'./data/videos/{VIDEO_NAME}_tilt.json'
OUTPUT_JSON_PATH = f'./data/videos/{VIDEO_NAME}_tilt_updated.json'
SCALE_FACTOR = 4
MIN_AREA = 1000

# Load JSON data
with open(INPUT_JSON_PATH, 'r') as json_file:
    json_content = json.load(json_file)

# Load video
cap = cv2.VideoCapture(f"./data/videos/{VIDEO_NAME}.mp4")

def update_json_for_frame(frame_idx, updated_boxes):
    """Update JSON content for a specific frame."""
    json_content[str(frame_idx)] = updated_boxes

def draw_boxes(frame, boxes, selected_idx=None):
    """Draw the boxes on the frame with an option to highlight the selected box."""
    for idx, box in enumerate(boxes):
        cx, cy, w, h, angle = box
        rect_points = cv2.boxPoints(((cx, cy), (w, h), angle))
        rect_points = np.intp(rect_points)
        
        if idx == selected_idx:
            cv2.polylines(frame, [rect_points], True, (0, 255, 0), 2)  # Green for selected box
            cv2.drawContours(frame, [rect_points], 0, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.polylines(frame, [rect_points], True, (0, 0, 255), 2)  # Red for other boxes

def interactive_adjustment(frame, boxes):
    """Interactive adjustment of bounding boxes using OpenCV."""
    def mouse_callback(event, x, y, flags, param):
        nonlocal start_x, start_y, drawing, selected_box, drag_mode, new_box
        if event == cv2.EVENT_LBUTTONDOWN:
            start_x, start_y = x, y
            drawing = True
            new_selected_box = None

            # Check if clicking on existing boxes
            for idx, box in enumerate(boxes):
                cx, cy, w, h, _ = box
                rect_points = cv2.boxPoints(((cx, cy), (w, h), 0))
                rect_points = np.intp(rect_points)
                if cv2.pointPolygonTest(rect_points, (x, y), False) >= 0:
                    new_selected_box = idx
                    break

            # Start drawing a new box
            if new_selected_box is None:
                new_box = [x, y, 0, 0, 0]
                boxes.append(new_box)
                selected_box = len(boxes) - 1
                drag_mode = 'resize'
            elif selected_box == new_selected_box and selected_box is not None:
                selected_box = None 
            else:
                selected_box = new_selected_box
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if selected_box is not None:
                    if drag_mode == 'resize':
                        new_x, new_y = x, y
                        if selected_box >= 0:
                            cx, cy, w, h, angle = boxes[selected_box]
                            new_w = max(new_x - cx, 1)
                            new_h = max(new_y - cy, 1)
                            boxes[selected_box] = [cx, cy, new_w, new_h, angle]
                    elif drag_mode == 'drag':
                        if selected_box >= 0:
                            cx, cy, w, h, angle = boxes[selected_box]
                            dx = x - start_x
                            dy = y - start_y
                            boxes[selected_box] = [cx + dx, cy + dy, w, h, angle]
                            start_x, start_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            drag_mode = None

            if selected_box is not None:
                selected = boxes[selected_box]

                if (selected[2] * selected[3] < 500):
                    boxes.pop(selected_box)
                    selected_box = None

    def handle_key_event(key):
        nonlocal selected_box
        if selected_box is not None:
            cx, cy, w, h, angle = boxes[selected_box]
            step = 5  # Pixel step for resizing

            if key == 27:  # ESC key to delete
                del boxes[selected_box]
                selected_box = None
            elif key == ord('r'):  # Rotate clockwise
                boxes[selected_box] = [cx, cy, w, h, (angle + 10) % 360]
            elif key == ord('c'):  # Rotate counterclockwise
                boxes[selected_box] = [cx, cy, w, h, (angle - 10) % 360]
            elif key == ord('w'):  # w key
                boxes[selected_box] = [cx, cy, w, h + step, angle]
            elif key == ord('d'):  # d key
                boxes[selected_box] = [cx, cy, w + step, h, angle]
            elif key == ord('a'):  # a key
                boxes[selected_box] = [cx, cy, w - step, h, angle]
            elif key == ord('s'):  # s key
                boxes[selected_box] = [cx, cy, w, h - step, angle]
            elif key == ord('i'): #up
                boxes[selected_box] = [cx, cy - step, w, h, angle]
            elif key == ord('k'): #down
                boxes[selected_box] = [cx, cy + step, w, h, angle]
            elif key == ord('j'): #left
                boxes[selected_box] = [cx - step, cy, w, h, angle]
            elif key == ord('l'): #right
                boxes[selected_box] = [cx + step, cy, w, h, angle]

    start_x, start_y = -1, -1
    drawing = False
    selected_box = None
    drag_mode = None
    new_box = None
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    while True:
        frame_copy = frame.copy()
        draw_boxes(frame_copy, boxes, selected_box)
        cv2.imshow('Frame', frame_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "EXIT"
        elif key == 32:  # Space key to proceed
            return boxes
        else:
            handle_key_event(key)

# Initialize
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    print(f"Beginning frame {frame_idx}")

    # Get bounding boxes for the current frame
    boxes_1 = json_content.get(str(frame_idx+1), [])
    boxes = []

    frame_height, frame_width = frame.shape[:2]
    frame = cv2.resize(frame, (frame_width // SCALE_FACTOR, frame_height // SCALE_FACTOR))

    for box in boxes_1:
        boxes.append([box[0] // SCALE_FACTOR, box[1] // SCALE_FACTOR, box[2] // SCALE_FACTOR, box[3] // SCALE_FACTOR, box[4]])

    # Draw current bounding boxes
    draw_boxes(frame.copy(), boxes)

    # Display the frame and allow interactive adjustments
    updated_boxes_1 = interactive_adjustment(frame, boxes)

    if updated_boxes_1 == "EXIT":
        break

    updated_boxes = []

    for box in updated_boxes_1:
        cx, cy, w, h, angle = box
        if w * h >= MIN_AREA:
            updated_boxes.append([cx * SCALE_FACTOR, cy * SCALE_FACTOR, w * SCALE_FACTOR, h * SCALE_FACTOR, angle])
    
    # Update JSON with the adjusted bounding boxes
    update_json_for_frame(frame_idx, updated_boxes)
    
    # Move to the next frame
    frame_idx += 1

# Release video capture
cap.release()

# Save updated JSON file
with open(OUTPUT_JSON_PATH, 'w') as json_file:
    json.dump(json_content, json_file, indent=4)

print(f'Updated JSON saved to {OUTPUT_JSON_PATH}')
