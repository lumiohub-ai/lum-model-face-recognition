import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


gt_path = 'output_mot_11.txt'
pred_path = 'output_11_processed_fps_recognition.txt'

# Load and prepare data 
gt_df = pd.read_csv(gt_path, names=['frame', 'track_id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility']) 
pred_df = pd.read_csv(pred_path, names=['frame', 'x', 'y', 'w', 'h', 'name'])

# Map track IDs to names in ground truth 
# 10 
# id_to_name = { 
#     1: 'Bahodir', 
#     2: 'Batkhuu', 
#     3: 'Mirsaid', 
#     4: 'Sarvar', 
#     5: 'Azamat', 
#     6: 'Maruf', 
#     7: 'Oybek', 
#     8: 'MuhammadAmin' 
# }

# 11
id_to_name = {
    1: 'Azamat',
    2: 'Oybek',
    3: 'Maruf',
    4: 'Bahodir',
    5: 'Sarvar',
    6: 'MuhammadAmin',
    7: 'Batkhuu',
    8: 'Mirsaid',
}


# Clean ground truth data 
gt_df = gt_df[['frame', 'track_id', 'x', 'y', 'w', 'h']] 
gt_df['name'] = gt_df['track_id'].map(id_to_name) 
gt_df = gt_df[['frame', 'x', 'y', 'w', 'h', 'name']]

# Save the cleaned ground truth data
gt_df.to_csv('gt_cleaned.csv', index=False)

# Lists to store ground truth and predicted names 
true_names = [] # Ground truth 
pred_names = [] # Predictions

# Calculate total_frames from the maximum frame number in both dataframes
total_frames = max(gt_df['frame'].max(), pred_df['frame'].max())

# Function to calculate IOU (Intersection over Union)
def calculate_iou(box1, box2):
    # Extract coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate coordinates of the intersection box
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Variables to count metrics
id_tp = 0  # Identity True Positive
id_fp = 0  # Identity False Positive
id_fn = 0  # Identity False Negative

# Lists to store ground truth and predicted names (only for matched detections)
true_names = []
pred_names = []

# Process each frame
for frame in range(1, total_frames + 1):
    # Get data for current frame
    frame_gt = gt_df[gt_df['frame'] == frame]
    frame_pred = pred_df[pred_df['frame'] == frame]
    
    # Skip if either ground truth or predictions are empty
    if frame_gt.empty or frame_pred.empty:
        if not frame_gt.empty:
            # All ground truth detections in this frame are missed (FN)
            id_fn += len(frame_gt)
            
        elif not frame_pred.empty:
            # All predicted detections in this frame are false positives (FP)
            id_fp += len(frame_pred)
        continue
    
    # Keep track of matched ground truth and predictions
    matched_gt = set()
    matched_pred = set()
    
    # Check each ground truth against each prediction
    for gt_idx, gt_row in frame_gt.iterrows():
        gt_box = [gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']]
        gt_name = gt_row['name']
        best_iou = 0
        best_pred_idx = None
        
        for pred_idx, pred_row in frame_pred.iterrows():
            if pred_idx in matched_pred:
                continue  # Skip already matched predictions
                
            pred_box = [pred_row['x'], pred_row['y'], pred_row['w'], pred_row['h']]
            pred_name = pred_row['name']
            
            # Calculate IOU between ground truth and prediction
            iou = calculate_iou(gt_box, pred_box)
            
            # Keep track of best match
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        
        # If we found a match with IOU > 0.5
        if best_iou > 0.5 and best_pred_idx is not None:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
            
            # Get the name of the matched prediction
            pred_name = frame_pred.loc[best_pred_idx, 'name']
            
            # Store names for confusion matrix
            true_names.append(gt_name)
            pred_names.append(pred_name)
            
            # Check if the predicted name is correct
            if pred_name == gt_name:
                id_tp += 1  # Identity True Positive
            else:
                id_fp += 1  # Identity False Positive
    
    # Count unmatched ground truth as false negatives
    id_fn += len(frame_gt) - len(matched_gt)
    
    # Count unmatched predictions as false positives
    id_fp += len(frame_pred) - len(matched_pred)


# Calculate ID precision, recall, F1, and accuracy
id_precision = id_tp / (id_tp + id_fp) if (id_tp + id_fp) > 0 else 0
id_recall = id_tp / (id_tp + id_fn) if (id_tp + id_fn) > 0 else 0
id_f1 = 2 * (id_precision * id_recall) / (id_precision + id_recall) if (id_precision + id_recall) > 0 else 0
id_accuracy = id_tp / (id_tp + id_fp + id_fn) if (id_tp + id_fp + id_fn) > 0 else 0

# Print results
print(f"ID-TP: {id_tp}")
print(f"ID-FP: {id_fp}")
print(f"ID-FN: {id_fn}")
print(f"ID Precision: {id_precision:.4f}")
print(f"ID Recall: {id_recall:.4f}")
print(f"ID F1 Score: {id_f1:.4f}")
print(f"ID Accuracy: {id_accuracy:.4f}")

if true_names and pred_names:
    # Get unique names (classes)
    unique_names = sorted(list(set(true_names) | set(pred_names)))

    # Calculate confusion matrix
    cm = confusion_matrix(true_names, pred_names, labels=unique_names)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Identity Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No matches found for confusion matrix.")