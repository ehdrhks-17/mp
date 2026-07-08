import cv2
import numpy as np
import mss
import time
from controller import tap_key, click_mouse, move_mouse

class CaptchaSolver:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.tracking_started = False
        self.target_center = None
        
        # V14: 동적 밝기 추적 (Blob Tracking) 상태 변수
        self.prev_x = 0
        self.prev_y = 0

    def find_initial_target(self, frame_bgr):
        """
        초기에 도형이 투명해지기 전(카운트다운 중 흰색 바탕도형 상태일 때) 찾아냅니다.
        """
        # 초기 락온 시엔 화면 전체를 블러 처리해서 부드럽게 타겟을 찾음
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Inpaint 대신 원본 프레임에서 흰색 바탕도형 찾기
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 흰색 바탕도형 찾기 (채도가 낮고 명도가 높은 색)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_white_box = None
        max_area = 0
        
        for c in contours_white:
            area = cv2.contourArea(c)
            if 1500 < area < 10000:
                x, y, w, h = cv2.boundingRect(c)
                extent = area / (w * h) if w * h > 0 else 0
                aspect = w / float(h) if h > 0 else 0
                
                # 좌측 게임 화면 내, 비율이 아주 엄격한 1:1에 가깝고(도형), 속이 어느 정도 꽉 찬(extent) 것만 필터링 (말풍선 등 제외)
                if x < 1280 and 0.95 < aspect < 1.05 and extent > 0.4:
                    if area > max_area:
                        max_area = area
                        best_white_box = (x, y, w, h)
                        
        if best_white_box:
            x, y, w, h = best_white_box
            self.prev_x = x + w // 2
            self.prev_y = y + h // 2
            self.tracking_started = True
            print("Target locked! Using V14 Dynamic Blob Tracking.")
            return True
            
        return None

    def solve_captcha(self):
        """
        거탐 팝업 시 호출됩니다.
        """
        print("[거탐 AI] 도형 탐색 시작")
        
        # 1. 초기 도형 찾기 (선명할 때)
        search_start = time.time()
        
        while time.time() - search_start < 10.0:
            screenshot = np.array(self.sct.grab(self.monitor))
            frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            if self.find_initial_target(frame_bgr):
                break
                
            time.sleep(0.1)
            
        if not self.tracking_started:
            print("[거탐 AI] 10초 동안 명확한 도형을 찾지 못했습니다.")
            return False
            
        # 3. 투명해지는 도형 실시간 추적 및 마우스 이동
        track_start = time.time()
        while time.time() - track_start < 25.0: # 거탐 지속시간 동안 유지
            screenshot = np.array(self.sct.grab(self.monitor))
            frame_original = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            # 1. 마우스 마스크 생성 (가장자리 흰/검 테두리까지 포함하도록 팽창)
            hsv = cv2.cvtColor(frame_original, cv2.COLOR_BGR2HSV)
            pink_mask = cv2.inRange(hsv, np.array([130, 30, 50]), np.array([175, 255, 255]))
            kernel = np.ones((7,7), np.uint8)
            mouse_mask = cv2.dilate(pink_mask, kernel, iterations=1)
            mouse_mask[:, 1280:] = 0 # 좌측 게임화면만
            
            # 회색조 변환 및 블러
            gray_frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)
            
            # 마우스가 있는 자리는 완전히 까맣게(0) 칠해버림 (계산에서 아예 배제)
            gray_frame[mouse_mask > 0] = 0
            
            # 이전 위치 주변 150x150 영역만 잘라냄
            search_size = 150
            sx1 = max(0, self.prev_x - search_size // 2)
            sy1 = max(0, self.prev_y - search_size // 2)
            sx2 = min(frame_original.shape[1], self.prev_x + search_size // 2)
            sy2 = min(frame_original.shape[0], self.prev_y + search_size // 2)
            
            roi = gray_frame[sy1:sy2, sx1:sx2]
            
            if roi.size > 0:
                # 동적으로 임계값 설정 (ROI 내의 0이 아닌 평균 밝기보다 20만큼 더 밝은 영역 추출)
                valid_pixels = roi[roi > 0]
                if len(valid_pixels) > 0:
                    mean_val = np.mean(valid_pixels)
                    thresh_val = mean_val + 20
                    _, thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    best_c = None
                    min_dist = float('inf')
                    
                    # ROI 중심(이전 타겟 위치)과 가장 가까운 큰 덩어리를 찾음
                    roi_center = (search_size//2, search_size//2)
                    for c in contours:
                        if cv2.contourArea(c) > 100: # 너무 작은 노이즈 무시
                            M = cv2.moments(c)
                            if M['m00'] > 0:
                                cx = int(M['m10']/M['m00'])
                                cy = int(M['m01']/M['m00'])
                                dist = (cx - roi_center[0])**2 + (cy - roi_center[1])**2
                                if dist < min_dist:
                                    min_dist = dist
                                    best_c = (cx, cy)
                                    
                    if best_c:
                        # 전체 좌표계로 변환
                        new_x = sx1 + best_c[0]
                        new_y = sy1 + best_c[1]
                        
                        # 부드럽게 이동 (Low-pass filter 적용하여 튀는 현상 방지)
                        self.prev_x = int(self.prev_x * 0.5 + new_x * 0.5)
                        self.prev_y = int(self.prev_y * 0.5 + new_y * 0.5)
                        
                        self.target_center = (self.prev_x, self.prev_y)
                        move_mouse(self.prev_x, self.prev_y)
                        print(f"Tracking... Center: ({self.prev_x}, {self.prev_y})")
            
            time.sleep(0.05) # 20 FPS
            
        print("[거탐 AI] 거탐 추적 사이클 종료.")
        return True

if __name__ == "__main__":
    solver = CaptchaSolver()
    solver.solve_captcha()
