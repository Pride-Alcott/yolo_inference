#run in the yolov5 virtual enviroment

import cv2
import glob
import os
import time
import csv

# Paths
model_path = '/home/pride/yolov5/best.pt'
images_folder = '/home/pride/Downloads/'
output_folder = '/home/pride/Downloads/inference_output/'
csv_log_path = '/home/pride/Downloads/inference_results.csv'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load your custom model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Get list of all images (jpg, png, jpeg)
image_paths = glob.glob(os.path.join(images_folder, '*.jpg')) + \
              glob.glob(os.path.join(images_folder, '*.jpeg')) + \
              glob.glob(os.path.join(images_folder, '*.png'))

print(f"Found {len(image_paths)} images.")

# Open CSV file for writing
with open(csv_log_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write CSV header
    writer.writerow(['Filename', 'Latency_ms', 'Width', 'Height'])

    # Inference loop
    for img_path in image_paths:
        print(f"Processing {img_path}...")

        # Read image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]  # Image dimensions

        # Run inference
        t1 = time.time()
        results = model(img)
        t2 = time.time()

        latency_ms = (t2 - t1) * 1000
        print(f"Inference time: {latency_ms:.2f} ms")

        # Render results
        annotated_frame = results.render()[0]

        # Save output image
        base_name = os.path.basename(img_path)
        save_path = os.path.join(output_folder, base_name)
        cv2.imwrite(save_path, annotated_frame)

        # Write results to CSV
        writer.writerow([base_name, f"{latency_ms:.2f}", w, h])

print("Inference and logging complete!")