import cv2
import numpy as np

def test_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # 1. Find initial white circle
    prev_x, prev_y = -1, -1
    prev_gray = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        
        # skip some initial frames if needed, or just look for the white circle
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_box = None
        for c in contours:
            area = cv2.contourArea(c)
            if 1000 < area < 10000:
                x, y, w, h = cv2.boundingRect(c)
                extent = area / (w * h)
                aspect = w / float(h)
                if 0.9 < aspect < 1.1 and extent > 0.4:
                    if area > max_area:
                        max_area = area
                        best_box = (x, y, w, h)
                        
        if best_box:
            x, y, w, h = best_box
            prev_x = x + w//2
            prev_y = y + h//2
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"Target locked at frame {frame_idx}: {prev_x}, {prev_y}")
            break
            
    if prev_gray is None:
        print("Initial target not found.")
        return
        
    # 2. Track using Local Motion (Mean-Shift on Difference)
    search_size = 120
    
    # Precompute Gaussian weights
    x_grid, y_grid = np.meshgrid(np.arange(search_size), np.arange(search_size))
    center = search_size / 2.0
    sigma = search_size / 4.0
    gaussian_weights = np.exp(-((x_grid - center)**2 + (y_grid - center)**2) / (2 * sigma**2))
    
    tracked_positions = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Absdiff
        diff = cv2.absdiff(gray, prev_gray)
        
        sx1 = max(0, prev_x - search_size // 2)
        sy1 = max(0, prev_y - search_size // 2)
        sx2 = min(frame.shape[1], prev_x + search_size // 2)
        sy2 = min(frame.shape[0], prev_y + search_size // 2)
        
        roi_diff = diff[sy1:sy2, sx1:sx2]
        
        if roi_diff.size > 0:
            # Threshold to remove noise
            _, roi_thresh = cv2.threshold(roi_diff, 10, 255, cv2.THRESH_BINARY)
            
            # Apply Gaussian weights if sizes match
            if roi_thresh.shape == gaussian_weights.shape:
                weighted_roi = roi_thresh.astype(np.float32) * gaussian_weights
                
                # Calculate center of mass
                M = cv2.moments(weighted_roi)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    new_x = sx1 + cx
                    new_y = sy1 + cy
                    
                    # Smooth update
                    prev_x = int(prev_x * 0.5 + new_x * 0.5)
                    prev_y = int(prev_y * 0.5 + new_y * 0.5)
            
        tracked_positions.append((prev_x, prev_y))
        prev_gray = gray
        
    print(f"Tracked {len(tracked_positions)} frames.")
    print(f"Last position: {tracked_positions[-1]}")

if __name__ == "__main__":
    test_tracking("captcha_vid.mp4")
