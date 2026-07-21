import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import face_recognition
import concurrent.futures
from itertools import combinations
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Face Database Augmentation')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing original face images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save augmented images')
    parser.add_argument('--augmentations_per_image', type=int, default=10, help='Number of augmentations per image')
    parser.add_argument('--detection_threshold', type=float, default=-1, 
                        help='Face detection confidence threshold. Set to negative value to skip detection (use when images are already verified to contain faces).')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--skip_verification', action='store_true', 
                        help='Skip face verification step completely (same as setting detection_threshold to negative value)')
    return parser.parse_args()

def create_augmentation_pipeline():
    """
    Create an augmentation pipeline with the specified techniques
    """
    return A.Compose([
        # Brightness adjustment
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.5),
        
        # Color jittering
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        
        # Gaussian noise
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
        
        # Gaussian blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        
        # Median blur
        A.MedianBlur(blur_limit=5, p=0.5),
        
        # Downsampling followed by upscaling (simulates low resolution)
        A.OneOf([
            A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=cv2.INTER_NEAREST, p=1.0),
        ], p=0.5),
        
        # Sharpening
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        
        # High contrast
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.3, 0.5), p=0.5),
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
        
        # Horizontal flip (Left/Right flip)
        A.HorizontalFlip(p=0.5),
    ])

def verify_face(image, detection_threshold=0.6):
    """
    Simply verify that there is a face in the image
    Returns the original image if a face is detected, None otherwise
    
    Note: This function is optional since images are already aligned,
    but provides a verification step to ensure the image contains a face
    """
    # Convert to RGB (face_recognition expects RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Using the faster HOG model since we're just verifying
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    
    if not face_locations:
        return None
        
    # If face is detected, return the original image
    return image

def process_image(args):
    """
    Process a single image and generate augmentations
    """
    image_path, output_dir, augmentations_per_image, detection_threshold, transform = args
    
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return 0
            
        # Optionally verify face presence (can be disabled if all images are guaranteed to contain faces)
        if detection_threshold > 0:
            face = verify_face(image, detection_threshold)
            if face is None:
                print(f"No face detected in: {image_path}")
                return 0
        else:
            face = image  # Skip detection if threshold is 0 or negative
            
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Generate augmentations using each technique individually
        augmentation_count = 0
        
        # Create individual augmentation functions for each technique
        augmentations = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0),  # Brightness
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),  # Color jittering
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),  # Gaussian noise
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),  # Gaussian blur
            A.MedianBlur(blur_limit=5, p=1.0),  # Median blur
            A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=cv2.INTER_LINEAR, p=1.0),  # Downsampling
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),  # Sharpening
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.5, p=1.0),  # High contrast
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),  # CLAHE
            A.HorizontalFlip(p=1.0),  # Horizontal flip
        ]
        
        # Apply each augmentation technique individually
        for i, aug in enumerate(augmentations):
            if i < augmentations_per_image:  # Only create up to the requested number
                augmented = aug(image=face)['image']
                aug_type = aug.__class__.__name__
                
                # Save augmented image with technique name directly in output folder
                aug_path = os.path.join(output_dir, f"{base_name}_{aug_type}.jpg")
                cv2.imwrite(aug_path, augmented)
                augmentation_count += 1
        
        # If we need more augmentations than individual techniques, use random combinations
        if augmentations_per_image > len(augmentations):
            # Apply random combinations for remaining augmentations
            for i in range(len(augmentations), augmentations_per_image):
                # Apply full augmentation pipeline with random combinations
                augmented = transform(image=face)['image']
                
                # Save augmented image directly in output folder
                aug_path = os.path.join(output_dir, f"{base_name}_combo_{i-len(augmentations)+1}.jpg")
                cv2.imwrite(aug_path, augmented)
                augmentation_count += 1
                
        return augmentation_count
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return 0

def main():
    args = parse_args()
    
    # If skip_verification flag is set, force detection_threshold to negative
    if args.skip_verification:
        args.detection_threshold = -1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No valid image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"{'Skipping' if args.detection_threshold < 0 else 'Using'} face verification")
    print(f"All augmented images will be saved directly in: {args.output_dir}")
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Create arguments for each image
    process_args = [(
        image_path, 
        args.output_dir, 
        args.augmentations_per_image, 
        args.detection_threshold, 
        transform
    ) for image_path in image_files]
    
    # Process images in parallel
    total_augmentations = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_image, process_args),
            total=len(image_files),
            desc="Augmenting faces"
        ))
        total_augmentations = sum(results)
    
    print(f"Augmentation complete! Generated {total_augmentations} augmented images.")
    print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()