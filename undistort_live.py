import cv2
import yaml
import numpy as np

# Load calibration data safely
with open('camera_calib.yaml', 'r') as f:
    raw_data = f.read()

# Fix tuples that PyYAML can't load
raw_data = raw_data.replace("!!python/tuple", "")
data = yaml.safe_load(raw_data)

# Convert lists back to numpy arrays
camera_matrix = np.array(data['camera_matrix'])
dist_coeffs = np.array(data['dist_coeff'])
image_width = data.get('image_width', 640)
image_height = data.get('image_height', 480)

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

# Get optimal new camera matrix
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (image_width, image_height), 1, (image_width, image_height)
)

print("Press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    # Undistort frame
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)

    # Combine for display
    combined = np.hstack((frame, undistorted))
    cv2.putText(combined, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(combined, "Undistorted", (image_width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Camera Calibration (Original vs Undistorted)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

