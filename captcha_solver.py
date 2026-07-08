import cv2
import numpy as np
import mss
import time
import keyboard
from controller import tap_key, click_mouse, move_mouse

class CaptchaSolver:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.tracking_started = False
        self.target_center = None
        
        # Hotkeys & GUI states
        self.is_paused = True  # 기본 시작 상태를 OFF(일시정지)로 설정
        self.needs_roi_selection = False
        self.roi_rect = None
        
        keyboard.add_hotkey('f10', self.toggle_pause)
        keyboard.add_hotkey('f8', self.trigger_roi_selection)
        
        # V14: 동적 밝기 추적 (Blob Tracking) 상태 변수
        self.prev_x = 0
        self.prev_y = 0

        self.prev_x = 0
        self.prev_y = 0

        # V16: 모션 트래킹 (프레임 차이 + 가우시안 윈도우) 상태 변수
        self.prev_gray = None
        self.search_size = 120
        # 미리 가우시안 가중치 맵을 계산해 둡니다.
        x_grid, y_grid = np.meshgrid(np.arange(self.search_size), np.arange(self.search_size))
        center = self.search_size / 2.0
        sigma = self.search_size / 4.0
        self.gaussian_weights = np.exp(-((x_grid - center)**2 + (y_grid - center)**2) / (2 * sigma**2))

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        print(f"[Hotkey] Tracking Paused: {self.is_paused}")
        
    def trigger_roi_selection(self):
        self.needs_roi_selection = True
        print("[Hotkey] Requested ROI selection.")

    def find_initial_target(self, frame_bgr):
        """
        초기에 도형이 투명해지기 전(카운트다운 중 흰색 바탕도형 상태일 때) 찾아냅니다.
        """
        # 노이즈를 줄여 윤곽선을 부드럽게 잡기 위해 블러 적용
        blurred_frame = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # 흰색 바탕도형 찾기 (채도가 낮고 명도가 높은 색)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_white_box = None
        max_area = 0
        
        for c in contours_white:
            area = cv2.contourArea(c)
            # 흰색 도형의 크기 필터링 (너무 작거나 слишком 큰 것 제외)
            if 1000 < area < 15000:
                x, y, w, h = cv2.boundingRect(c)
                extent = area / (w * h) if w * h > 0 else 0
                aspect = w / float(h) if h > 0 else 0
                
                in_roi = True
                if self.roi_rect:
                    rx, ry, rw, rh = self.roi_rect
                    if not (rx <= x and x+w <= rx+rw and ry <= y and y+h <= ry+rh):
                        in_roi = False
                
                # 비율 조건을 기존 0.95~1.05에서 0.8~1.25로 대폭 완화 (찌그러지거나 픽셀 깨짐 허용)
                if in_roi and 0.8 < aspect < 1.25 and extent > 0.4:
                    if area > max_area:
                        max_area = area
                        best_white_box = (x, y, w, h)
                        
        if best_white_box:
            x, y, w, h = best_white_box
            self.prev_x = x + w // 2
            self.prev_y = y + h // 2
            self.tracking_started = True
            print("Target locked! Using V16 Gaussian Motion Tracking.")
            return True
            
        return None

    def update_motion_tracker(self, frame_original):
        """
        V16: 이전 프레임과 현재 프레임의 차이(Motion)를 계산하고,
             가우시안 가중치를 적용하여 중앙 타깃의 움직임만 추적합니다.
        """
        if self.prev_gray is None:
            return False
            
        # 1. 마우스가 움직이는 것을 모션으로 착각하지 않도록 분홍색 커서 가리기
        hsv = cv2.cvtColor(frame_original, cv2.COLOR_BGR2HSV)
        pink_mask = cv2.inRange(hsv, np.array([130, 30, 50]), np.array([175, 255, 255]))
        kernel = np.ones((7,7), np.uint8)
        mouse_mask = cv2.dilate(pink_mask, kernel, iterations=1)
        
        gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        gray[mouse_mask > 0] = 0  # 마우스 커서 부분 블랙아웃
        
        # 2. 프레임 차이(움직임) 계산
        diff = cv2.absdiff(gray, self.prev_gray)
        
        # 3. 현재 타깃 중심 주변 영역(ROI)만 잘라내기
        sx1 = max(0, self.prev_x - self.search_size // 2)
        sy1 = max(0, self.prev_y - self.search_size // 2)
        sx2 = min(frame_original.shape[1], self.prev_x + self.search_size // 2)
        sy2 = min(frame_original.shape[0], self.prev_y + self.search_size // 2)
        
        roi_diff = diff[sy1:sy2, sx1:sx2]
        
        tracked = False
        if roi_diff.size > 0:
            # 잔상에서 확실한 움직임(픽셀값 15 이상 변화)만 추출
            _, roi_thresh = cv2.threshold(roi_diff, 15, 255, cv2.THRESH_BINARY)
            
            # 잘라낸 크기가 가우시안 윈도우 크기와 일치할 때만 가중치 적용 (화면 가장자리 예외 처리)
            if roi_thresh.shape == self.gaussian_weights.shape:
                weighted_roi = roi_thresh.astype(np.float32) * self.gaussian_weights
                
                # 무게 중심(Center of Mass) 계산
                M = cv2.moments(weighted_roi)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    new_x = sx1 + cx
                    new_y = sy1 + cy
                    
                    # 마우스 부드럽게 이동 (50%만 반영하여 떨림 방지)
                    self.prev_x = int(self.prev_x * 0.5 + new_x * 0.5)
                    self.prev_y = int(self.prev_y * 0.5 + new_y * 0.5)
                    tracked = True
                    
        # 현재 프레임을 다음 프레임 비교를 위해 저장
        # 단, 마우스가 지워진 부분 때문에 다음 프레임에서 구멍이 파이는 것을 막기 위해 원본 그레이를 저장
        clean_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        self.prev_gray = clean_gray
        
        return tracked

    def _draw_gui(self, frame, state_text):
        if self.roi_rect:
            x, y, w, h = self.roi_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, "ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        color = (0, 0, 255) if self.is_paused else (0, 255, 0)
        status = "PAUSED" if self.is_paused else state_text
        cv2.putText(frame, f"State: {status} | F8: Set ROI | F10: Play/Pause", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Captcha AI - Detection View", frame)

    def run(self):
        """
        GUI 기반 메인 루프. F10으로 끄고 켜며, F8로 영역을 지정합니다.
        """
        print("[거탐 AI] 봇 실행됨. F8: 감지 영역 지정, F10: 시작/일시정지")
        cv2.namedWindow("Captcha AI - Detection View", cv2.WINDOW_NORMAL)
        
        while True:
            if self.needs_roi_selection:
                screenshot = np.array(self.sct.grab(self.monitor))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                print("[GUI] 드래그하여 감지 영역을 설정하고 SPACE 또는 ENTER를 누르세요.")
                roi = cv2.selectROI("Captcha AI - Detection View", frame, showCrosshair=True, fromCenter=False)
                if roi[2] > 0 and roi[3] > 0:
                    self.roi_rect = roi
                    print(f"[GUI] 감지 영역 설정 완료: {self.roi_rect}")
                self.needs_roi_selection = False

            if self.is_paused:
                screenshot = np.array(self.sct.grab(self.monitor))
                vis_frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                self._draw_gui(vis_frame, "PAUSED")
                if cv2.waitKey(50) & 0xFF == 27:
                    break
                continue

            # --- SEARCH PHASE ---
            self.tracking_started = False
            self.prev_gray = None
            
            screenshot = np.array(self.sct.grab(self.monitor))
            frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            vis_frame = frame_bgr.copy()
            
            if self.find_initial_target(frame_bgr):
                self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            if not self.tracking_started:
                self._draw_gui(vis_frame, "SEARCHING")
                if cv2.waitKey(50) & 0xFF == 27:
                    break
                continue
                
            # --- TRACKING PHASE ---
            track_start = time.time()
            while time.time() - track_start < 25.0: # 거탐 지속시간 동안 유지
                if self.is_paused or self.needs_roi_selection:
                    break # F10이나 F8을 누르면 추적 중단하고 메인 루프로 복귀
                    
                screenshot = np.array(self.sct.grab(self.monitor))
                frame_original = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                vis_frame = frame_original.copy()
                
                tracked = self.update_motion_tracker(frame_original)
                        
                if tracked:
                    self.target_center = (self.prev_x, self.prev_y)
                    move_mouse(self.prev_x, self.prev_y)
                    # 시각화: V16 모션 타깃 중심 및 추적 반경(가우시안 윈도우) 표시
                    cv2.circle(vis_frame, (self.prev_x, self.prev_y), self.search_size // 2, (0, 255, 255), 1)
                    cv2.circle(vis_frame, (self.prev_x, self.prev_y), 5, (0, 0, 255), -1)
                    print(f"[V16] Tracking... Center: ({self.prev_x}, {self.prev_y})")
                else:
                    # 1. 마우스 마스크 생성
                    hsv = cv2.cvtColor(frame_original, cv2.COLOR_BGR2HSV)
                    pink_mask = cv2.inRange(hsv, np.array([130, 30, 50]), np.array([175, 255, 255]))
                    kernel = np.ones((7,7), np.uint8)
                    mouse_mask = cv2.dilate(pink_mask, kernel, iterations=1)
                    
                    if self.roi_rect:
                        rx, ry, rw, rh = self.roi_rect
                        mask_roi = np.zeros_like(mouse_mask)
                        mask_roi[ry:ry+rh, rx:rx+rw] = 255
                        mouse_mask = cv2.bitwise_and(mouse_mask, mask_roi)
                    else:
                        mouse_mask[:, 1280:] = 0 # 좌측 게임화면만
                    
                    gray_frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)
                    gray_frame[mouse_mask > 0] = 0
                    
                    search_size = 150
                    sx1 = max(0, self.prev_x - search_size // 2)
                    sy1 = max(0, self.prev_y - search_size // 2)
                    sx2 = min(frame_original.shape[1], self.prev_x + search_size // 2)
                    sy2 = min(frame_original.shape[0], self.prev_y + search_size // 2)
                    
                    roi = gray_frame[sy1:sy2, sx1:sx2]
                    
                    if roi.size > 0:
                        valid_pixels = roi[roi > 0]
                        if len(valid_pixels) > 0:
                            mean_val = np.mean(valid_pixels)
                            thresh_val = mean_val + 20
                            _, thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            best_c = None
                            min_dist = float('inf')
                            roi_center = (search_size//2, search_size//2)
                            
                            for c in contours:
                                if cv2.contourArea(c) > 100:
                                    M = cv2.moments(c)
                                    if M['m00'] > 0:
                                        cx = int(M['m10']/M['m00'])
                                        cy = int(M['m01']/M['m00'])
                                        dist = (cx - roi_center[0])**2 + (cy - roi_center[1])**2
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_c = (cx, cy)
                                            
                            if best_c:
                                new_x = sx1 + best_c[0]
                                new_y = sy1 + best_c[1]
                                self.prev_x = int(self.prev_x * 0.5 + new_x * 0.5)
                                self.prev_y = int(self.prev_y * 0.5 + new_y * 0.5)
                                self.target_center = (self.prev_x, self.prev_y)
                                move_mouse(self.prev_x, self.prev_y)
                                
                                # 시각화: V14 폴백 추적
                                cv2.circle(vis_frame, (self.prev_x, self.prev_y), 5, (255, 0, 0), -1)
                
                self._draw_gui(vis_frame, "TRACKING")
                if cv2.waitKey(30) & 0xFF == 27:
                    return
            
            print("[거탐 AI] 거탐 추적 사이클 종료. 다시 탐색 상태로 돌아갑니다.")

if __name__ == "__main__":
    solver = CaptchaSolver()
    solver.run()
