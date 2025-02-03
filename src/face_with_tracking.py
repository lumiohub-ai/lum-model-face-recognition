# import torch
# import cv2
# from facenet_pytorch import InceptionResnetV1
# from ultralytics import YOLO
# # from sklearn.metrics.pairwise import cosine_similarity
# face_embeddings_path = 'data/embeddings'
# face_crops_path = 'data/images'

# # Initialize YOLO and FaceNet
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# yolo_model = YOLO("yolov8m-face.pt")
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# metric = nn_matching.NearestNeighborDistanceMetric(
#         'cosine',
#         0.8,
#         100)

# tracker = Tracker(metric)


# cap = cv2.VideoCapture('rtsp://admin:hbai2024@172.30.1.44:554/Streaming/Channels/1')

# name_to_track_map = {}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     results = yolo_model.predict(source=frame, conf=0.5, max_det=1)

#     detections = []
#     boxes = results[0].boxes.data.cpu().numpy()
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box[:4])
#         width = x2 - x1
#         height = y2 - y1
#         conf = box[4]
#         bbox = (x1, y1, width, height)

#         face = frame[y1:y2, x1:x2]

#         face_tensor = torch.tensor(cv2.resize(face, (160, 160))).permute(2, 0, 1).float().to(device)
#         face_tensor = (face_tensor / 255.0).unsqueeze(0)

#         if face.shape[0] < 30 or face.shape[1] < 30:
#             continue

#         with torch.no_grad():
#             emb = resnet(face_tensor).cpu().numpy().flatten()

#         detection = Detection(bbox, conf, emb)
#         detections.append(detection)

#     tracker.predict()
#     tracker.update(detections)

#     for track in tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         name = None
#         if track.track_id not in name_to_track_map:
#             name = input("Enter name: ")
#             name_to_track_map[track.track_id] = name
#         else:
#             name = name_to_track_map[track.track_id]

#         bbox = track.to_tlwh()
#         x, y, w, h = map(int, bbox)

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # display name on the bounding box
#         cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()