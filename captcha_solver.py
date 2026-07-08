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
        
        # V12: 도넛 형태의 템플릿 매칭 상태 변수
        self.template = None
        self.tm_mask = None
        self.prev_x = 0
        self.prev_y = 0
        self.w = 0
        self.h = 0

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
            self.w, self.h = max(w, h), max(w, h) # 완벽한 정사각형으로 맞춤
            self.prev_x = x + self.w // 2
            self.prev_y = y + self.h // 2
            
            # 초기 타겟을 템플릿으로 저장
            self.template = gray_frame[y:y+self.h, x:x+self.w].copy()
            
            # 도넛 마스크 생성 (가운데 60% 영역은 계산에서 완전히 무시 = 마우스가 지나가도 인식 안 함)
            self.tm_mask = np.zeros((self.h, self.w), dtype=np.uint8)
            center = (self.w // 2, self.h // 2)
            outer_radius = min(self.w, self.h) // 2
            inner_radius = int(outer_radius * 0.6) 
            cv2.circle(self.tm_mask, center, outer_radius, 255, -1)
            cv2.circle(self.tm_mask, center, inner_radius, 0, -1)
            
            self.tracking_started = True
            print("Target locked! Using Donut Template Matching.")
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
            gray_frame = cv2.cvtColor(frame_original, cv2.COLOR_BGRA2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            
            if self.template is not None and self.tm_mask is not None:
                # 이전 위치 주변 150x150 픽셀 영역에서만 탐색
                search_size = 150
                sx1 = max(0, self.prev_x - search_size // 2)
                sy1 = max(0, self.prev_y - search_size // 2)
                sx2 = min(frame_original.shape[1], self.prev_x + search_size // 2)
                sy2 = min(frame_original.shape[0], self.prev_y + search_size // 2)
                
                search_region = gray_frame[sy1:sy2, sx1:sx2]
                
                if search_region.shape[0] >= self.h and search_region.shape[1] >= self.w:
                    best_score = -1
                    best_loc = None
                    
                    # V13: 도형이 회전하는 것을 감안하여 36개 각도(0~350도)로 돌려가며 가장 똑같은 모양을 찾음
                    for angle in range(0, 360, 10):
                        M = cv2.getRotationMatrix2D((self.w // 2, self.h // 2), angle, 1.0)
                        rotated_template = cv2.warpAffine(self.template, M, (self.w, self.h))
                        rotated_mask = cv2.warpAffine(self.tm_mask, M, (self.w, self.h))
                        
                        # 마스크를 적용한 템플릿 매칭 (가운데가 뚫린 도넛 형태만 비교)
                        res = cv2.matchTemplate(search_region, rotated_template, cv2.TM_CCORR_NORMED, mask=rotated_mask)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > best_score:
                            best_score = max_val
                            best_loc = max_loc
                    
                    if best_loc is not None:
                        # 새로운 중심 좌표 계산
                        best_x = sx1 + best_loc[0]
                        best_y = sy1 + best_loc[1]
                        
                        self.prev_x = best_x + self.w // 2
                        self.prev_y = best_y + self.h // 2
                        self.target_center = (self.prev_x, self.prev_y)
                        
                        # 마우스로 이동
                        move_mouse(self.prev_x, self.prev_y)
                        print(f"Tracking... Center: ({self.prev_x}, {self.prev_y}), Score: {best_score:.2f}")
            
            time.sleep(0.05) # 20 FPS
            
        print("[거탐 AI] 거탐 추적 사이클 종료.")
        return True

if __name__ == "__main__":
    solver = CaptchaSolver()
    solver.solve_captcha()
