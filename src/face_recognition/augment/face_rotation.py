import os
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Face Warping Tool (Optimized)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing face images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save warped images')
    parser.add_argument('--intensity', type=float, default=0.2, help='Warping intensity (0.1-0.5)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    return parser.parse_args()

def create_warp_maps(shape, direction, intensity=0.2):
    """
    Create warping maps for a given direction.
    """
    h, w = shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = x.astype(np.float32)
    map_y = y.astype(np.float32)
    
    # Calculate distance from center (normalized)
    dist_x = (x - center_x) / (w/2)
    dist_y = (y - center_y) / (h/2)
    dist = np.sqrt(dist_x**2 + dist_y**2)
    
    # Create falloff mask (1 at center, 0 at edges)
    mask = np.clip(1.0 - dist, 0, 1)
    
    # Apply different warping based on direction
    if direction == 'left':
        # Create horizontal gradient (0 at left, 1 at right)
        gradient = x / w
        # Apply warping: more shifting on the right side of the face
        shift = intensity * w * gradient * mask
        map_x += shift
        
    elif direction == 'right':
        # Create horizontal gradient (1 at left, 0 at right)
        gradient = 1.0 - (x / w)
        # Apply warping: more shifting on the left side of the face
        shift = -intensity * w * gradient * mask
        map_x += shift
        
    elif direction == 'up':
        # Create vertical gradient (0 at top, 1 at bottom)
        gradient = y / h
        # Apply warping: more shifting at the bottom of the face
        shift = intensity * h * gradient * mask
        map_y += shift
        
    elif direction == 'down':
        # Create vertical gradient (1 at top, 0 at bottom)
        gradient = 1.0 - (y / h)
        # Apply warping: more shifting at the top of the face
        shift = -intensity * h * gradient * mask
        map_y += shift
    
    return map_x, map_y

def warp_face(image, direction, intensity=0.2):
    """
    Apply warping to make the face appear to be looking in different directions.
    """
    # Create warping maps
    map_x, map_y = create_warp_maps(image.shape, direction, intensity)
    
    # Apply the warping
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return warped

def process_image(args):
    """Process a single image with face warping."""
    image_path, output_dir, intensity = args
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return 0
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Define warping directions
        directions = ['left', 'right', 'up', 'down']
        
        warp_count = 0
        # Apply different warpings
        for direction in directions:
            # Warp face
            warped_image = warp_face(image, direction, intensity)
            
            # Save warped image
            output_path = os.path.join(output_dir, f"{base_name}_{direction}.jpg")
            cv2.imwrite(output_path, warped_image)
            warp_count += 1
        
        return warp_count
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return 0\
        

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files
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
    print(f"All warped images will be saved directly in: {args.output_dir}")
    print(f"Using warping intensity: {args.intensity}")
    
    # Create process arguments
    process_args = [(image_path, args.output_dir, args.intensity) for image_path in image_files]
    
    # Process images in parallel
    warp_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_image, process_args),
            total=len(image_files),
            desc="Warping faces"
        ))
        warp_count = sum(results)
    
    print(f"Face warping complete! Generated {warp_count} warped images.")
    print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()