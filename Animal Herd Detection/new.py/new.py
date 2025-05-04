import cv2
import torch
import numpy as np
import folium
from datetime import datetime


model = torch.hub.load('yolov5', 'yolov5s', source='local')  
animal_classes = ['cow', 'horse', 'sheep', 'dog', 'cat', 'bird']  


map_center = [28.6139, 77.2090]  
herd_map = folium.Map(location=map_center, zoom_start=5)


def simulate_gps(index):
    return [28.6 + (index * 0.01), 77.2 + (index * 0.01)]


DIST_THRESHOLD = 100
HERD_MIN_SIZE = 3


cap = cv2.VideoCapture('input/your_video.mp4')
frame_id = 0
herd_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    
    results = model(frame)
    detections = results.pandas().xyxy[0]

    
    animals = detections[detections['name'].isin(animal_classes)]

    
    centers = []
    for idx, row in animals.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        centers.append((cx, cy))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    herd_centers = []
    for i, c1 in enumerate(centers):
        group = [c1]
        for j, c2 in enumerate(centers):
            if i != j:
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                if dist < DIST_THRESHOLD:
                    group.append(c2)
        if len(group) >= HERD_MIN_SIZE:
            for c in group:
                if c not in herd_centers:
                    herd_centers.append(c)
            
            gps = simulate_gps(herd_count)
            herd_count += 1
            folium.Marker(
                location=gps,
                popup=f"Herd detected: {len(group)} animals at {datetime.now().strftime('%H:%M:%S')}",
                icon=folium.Icon(color='red')
            ).add_to(herd_map)

    
    for c in herd_centers:
        cv2.circle(frame, c, 5, (0, 0, 255), -1)

    cv2.imshow("Herd Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


herd_map.save('herd_map.html')
cap.release()
cv2.destroyAllWindows()
