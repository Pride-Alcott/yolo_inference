import cv2
import os

output_dir = "calibration_images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("Press SPACE to capture an image, ESC to quit.")

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Capture Calibration Images", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("Exiting...")
        break
    elif key % 256 == 32:  # SPACE
        filename = os.path.join(output_dir, f"image_{i:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}")
        i += 1

cap.release()
cv2.destroyAllWindows()
