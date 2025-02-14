import cv2
from extract_annotations.draw_polygons import PolygonManager
from extract_annotations.draw_rectangle import RectangleManager
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

def main():
    video_path = "rtsp://admin:hbai2024@172.30.1.87:554/Streaming/Channels/1"
    output_video_path = "/home/hbvision/mirsaid/smart-office/videos/IMG_7121_output.mp4"
    cap = cv2.VideoCapture(video_path)

    annotations = polygon_manager.set_polygons(cap, add_polygon=True) # if add_polygon is True, you can add 
    # annotations = rectangle_manager.set_rectangles(cap, add_rectangle=True)
    # new polygons to the existing ones, 
    # otherwise, you can only view the existing polygons

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter for output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    detector = YOLO("/home/hbvision/mirsaid/smart-office/ablation_17x.pt")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections and tracking info from person model
        results = detector.track(source=frame, persist=True, classes=[0])
        # frame = polygon_manager.draw_polygons(frame)

        # Store polygons and their classification status
        polygons = []
        
        # Process annotation polygons
        for line in annotations:
            if line.startswith("#") or not line:
                continue
            values = line.split()
            class_id = int(values[0])
            coords = [float(val) for val in values[1:]]
            
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * width)
                y = int(coords[i + 1] * height)
                points.append((x, y))
            
            polygons.append({"points": points, "label": "Not at Desk", "class_id": class_id})
        
        for result in results:
            for box in result.boxes:
                # class_id = int(box.cls[0])
                # if class_id != 0:
                #     continue  # Only process "person" detections

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                # Calculate center point of bounding box for polygon check
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                person_point = Point(center_x, center_y)

                # Update polygon label if person is using phone
                for poly in polygons:
                    polygon_shape = Polygon(poly["points"])
                    if polygon_shape.contains(person_point):
                        poly["label"] = "At Desk"
                        break

                # Draw person bounding box and confidence label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
        for poly in polygons:
            leftmost_point = min(poly["points"], key=lambda point: point[0])
            label_position = (leftmost_point[0] - 50, leftmost_point[1])
            
            # Set the color based on the label status
            if poly["label"] == "At Desk":
                label_color = (0, 255, 0)  # Green
            else:
                label_color = (0, 0, 255)  # Red

            # Draw the classification label at the determined position with the specified color
            cv2.putText(frame, poly["label"], label_position, 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 1)
            # cv2.polylines(frame, [np.array(poly["points"])], True, label_color, 2)
            # cv2.putText(frame, str(poly["class_id"]), label_position,   cv2.FONT_HERSHEY_SIMPLEX, 1, label_color
            #             , 1)
        
            
        cv2.imshow("Frame", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        
if __name__ == "__main__":

    polygon_manager = PolygonManager()
    rectangle_manager = RectangleManager()

    main()