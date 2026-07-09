import cv2
import numpy as np
import mss
import mouse
import time

class CaptchaSolver:
    def __init__(self, debug_callback=None):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.debug_callback = debug_callback
        
        self.state = "SEARCHING"
        self.kalman = None
        self.last_box_size = (60, 60)
        self.prev_gray = None
        
    def _preprocess(self, frame_bgr):
        # Motion Compensated Frame Differencing (The Ultimate Preprocessing)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_f = np.float32(gray)
        
        if self.prev_gray is None or self.prev_gray.shape != gray_f.shape:
            self.prev_gray = gray_f
            return np.zeros_like(gray)
            
        # 1. Calculate background shift (Phase Correlation)
        shift, _ = cv2.phaseCorrelate(self.prev_gray, gray_f)
        dx, dy = shift
        
        # 2. Warp previous frame to align with current frame
        rows, cols = gray.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_prev = cv2.warpAffine(self.prev_gray, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # 3. Absolute difference to isolate ONLY moving objects
        diff = cv2.absdiff(gray_f, aligned_prev)
        diff_8u = np.clip(diff, 0, 255).astype(np.uint8)
        
        # 4. Threshold and morphology to clean up
        # Lower threshold to 15 to catch fainter distortions
        _, thresh = cv2.threshold(diff_8u, 15, 255, cv2.THRESH_BINARY)
        
        # Heavy CLOSE to bridge broken shapes, followed by dilation
        kernel = np.ones((9,9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # 5. Filter out the Pink Mouse Pointer
        # The mouse pointer is a solid object that creates massive motion blobs.
        # We find its pink color (H: 140-170) and mask it out completely.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([135, 50, 150])
        upper_pink = np.array([175, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Dilate the pink mask heavily to cover the white borders of the crosshair
        pink_kernel = np.ones((25, 25), np.uint8)
        pink_mask = cv2.dilate(pink_mask, pink_kernel, iterations=1)
        
        # Erase the mouse pointer from the motion mask
        thresh[pink_mask > 0] = 0
        
        self.prev_gray = gray_f
        
        return thresh

    def _init_kalman(self, x, y):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        # Constant Velocity Model
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],
                                                [0,1,0,0],
                                                [0,0,5,0],
                                                [0,0,0,5]], np.float32) * 0.05
        self.kalman.measurementNoiseCov = np.array([[1,0],
                                                    [0,1]], np.float32) * 1e-1
        
        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    def find_initial_target(self, frame_bgr):
        # Look for the solid white star initially
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        max_area = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if 300 < area < 5000:
                x, y, w, h = cv2.boundingRect(c)
                aspect = float(w) / h
                extent = area / (w*h)
                if 0.7 < aspect < 1.3 and extent > 0.3:
                    if area > max_area:
                        max_area = area
                        best_rect = (x, y, w, h)
                        
        return best_rect

    def process_frame(self, frame_bgr, screen_offset_x, screen_offset_y):
        debug_img = frame_bgr.copy()
        target_center_screen = None
        
        # Get motion mask
        motion_mask = self._preprocess(frame_bgr)
        preprocessed_3c = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        
        if self.state == "SEARCHING":
            rect = self.find_initial_target(frame_bgr)
            if rect:
                x, y, w, h = rect
                cx, cy = x + w/2, y + h/2
                self._init_kalman(cx, cy)
                self.last_box_size = (w, h)
                self.state = "TRACKING"
                
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, "LOCKED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif self.state == "TRACKING":
            # 1. Predict position
            pred = self.kalman.predict()
            pred_x, pred_y = float(pred[0][0]), float(pred[1][0])
            
            # 2. Find all moving objects in the motion mask
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_dist = float('inf')
            best_meas = None
            best_rect = None
            
            # Track coasting frames to dynamically expand search radius if lost
            if not hasattr(self, 'coast_frames'):
                self.coast_frames = 0
                
            dynamic_radius = 60 + (self.coast_frames * 20)
            if dynamic_radius > 150:
                dynamic_radius = 150
            
            # Find the moving object closest to our prediction
            for c in contours:
                area = cv2.contourArea(c)
                if area > 40:  # Lowered to catch broken shapes
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Structural Filter: "테두리로만 이루어진걸 찾아야함"
                    roi_thresh = motion_mask[y:y+h, x:x+w]
                    density = cv2.countNonZero(roi_thresh) / (w * h)
                    
                    # If it's a solid blob (density > 0.6), ignore it
                    if density > 0.6:
                        continue
                        
                    cx, cy = x + w/2, y + h/2
                    
                    dist = np.hypot(cx - pred_x, cy - pred_y)
                    # Use dynamic radius
                    if dist < best_dist and dist < dynamic_radius:
                        best_dist = dist
                        best_meas = (cx, cy)
                        best_rect = (x, y, w, h)
            
            if best_meas:
                self.coast_frames = 0 # Reset coasting
                # Target found! Update Kalman filter
                meas_cx, meas_cy = best_meas
                measurement = np.array([[np.float32(meas_cx)], [np.float32(meas_cy)]])
                self.kalman.correct(measurement)
                final_cx, final_cy = meas_cx, meas_cy
                
                if best_rect:
                    _, _, tw, th = best_rect
                    self.last_box_size = (tw, th)
                    
                tw, th = self.last_box_size
                cv2.rectangle(debug_img, (int(final_cx-tw/2), int(final_cy-th/2)), 
                              (int(final_cx+tw/2), int(final_cy+th/2)), (0, 255, 0), 2)
                cv2.putText(debug_img, "TRACKING", (int(final_cx-tw/2), int(final_cy-th/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Target lost (occlusion/stopped). Coasting!
                self.coast_frames += 1
                final_cx, final_cy = pred_x, pred_y
                tw, th = self.last_box_size
                cv2.rectangle(debug_img, (int(final_cx-tw/2), int(final_cy-th/2)), 
                              (int(final_cx+tw/2), int(final_cy+th/2)), (0, 165, 255), 2)
                cv2.putText(debug_img, "COASTING", (int(final_cx-tw/2), int(final_cy-th/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
            # Draw prediction dot
            cv2.circle(debug_img, (int(pred_x), int(pred_y)), 5, (0, 0, 255), -1)
            
            target_center_screen = (int(final_cx) + screen_offset_x, int(final_cy) + screen_offset_y)

            # Optional debug: Draw all detected moving objects in red to show they are ignored
            for c in contours:
                if cv2.contourArea(c) > 80:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(preprocessed_3c, (x, y), (x+w, y+h), (0, 0, 255), 1)

        if target_center_screen:
            # 즉각적으로 마우스 이동
            mouse.move(target_center_screen[0], target_center_screen[1], absolute=True, duration=0)
            
        if self.debug_callback:
            self.debug_callback(debug_img, preprocessed_3c)
            
        return debug_img
