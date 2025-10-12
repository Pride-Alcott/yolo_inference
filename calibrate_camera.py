import cv2
import numpy as np
import glob
import yaml

# === Configuration ===
chessboard_size = (9, 6)
square_size = 0.017  # meters
camera_name = "Microsoft Webcam"

# === Prepare object points ===
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob('calibration_images/*.jpg')

print(f"Found {len(images)} calibration images")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"❌ Chessboard not detected in {fname}")

cv2.destroyAllWindows()

# === Calibrate ===
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# === Compute Reprojection Error ===
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
mean_error /= len(objpoints)

# === Save results ===
data = {
    'camera_matrix': mtx.tolist(),
    'dist_coeff': dist.tolist(),
    'reprojection_error': float(mean_error),
    'num_images': len(images),
    'chessboard_size': chessboard_size,
    'square_size': square_size
}
with open("camera_calib.yaml", "w") as f:
    yaml.dump(data, f)

# === Display formatted results ===
fx, fy = mtx[0, 0], mtx[1, 1]
cx, cy = mtx[0, 2], mtx[1, 2]
k1, k2, p1, p2, k3 = dist.ravel()[:5]

print("\n" + "="*60)
print(f"5.1: Camera Calibration Results for {camera_name}")
print("="*60)
print(f"Image Resolution: {gray.shape[1]}×{gray.shape[0]} px")
print(f"Focal Length (fx, fy): ({fx:.2f}, {fy:.2f}) pixels")
print(f"Principal Point (cx, cy): ({cx:.2f}, {cy:.2f}) pixels")
print(f"Radial Distortion (k1, k2, k3): ({k1:.3f}, {k2:.3f}, {k3:.3f})")
print(f"Tangential Distortion (p1, p2): ({p1:.3f}, {p2:.3f})")
print(f"Reprojection Error (RMS): {mean_error:.3f} pixels")
print(f"Number of Calibration Images: {len(images)}")
print(f"Chessboard Pattern Size: {chessboard_size[0]}×{chessboard_size[1]} corners")
print(f"Square Size: {square_size*1000:.0f} mm")
print("Calibration file saved as 'camera_calib.yaml'")
print("="*60)
