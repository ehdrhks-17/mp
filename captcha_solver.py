import cv2
import numpy as np
import mss
import time
from controller import tap_key, click_mouse, move_mouse

class CaptchaSolver:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

    def find_initial_target(self, frame_bgr):
        """
        초기에 도형이 투명해지기 전(카운트다운 중 흰색 바탕도형 상태일 때) 찾아냅니다.
        """
        # 마우스 커서의 핑크색 '테두리'만 찾아내서 얇게 지움 (투명 타겟 중심부 보존)
        hsv_pre = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        pink_mask = cv2.inRange(hsv_pre, np.array([130, 30, 50]), np.array([175, 255, 255]))
        
        if cv2.countNonZero(pink_mask) > 0:
            kernel = np.ones((3,3), np.uint8)
            dilated_mask = cv2.dilate(pink_mask, kernel, iterations=1)
            # 메이플 게임 화면(좌측)에 있는 마우스만 지움
            dilated_mask[:, 1280:] = 0
            frame_bgr = cv2.inpaint(frame_bgr, dilated_mask, 3, cv2.INPAINT_TELEA)

        # Inpaint 처리된 이미지에서 흰색 바탕도형 찾기
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
            # 락온! 타겟 크기에 맞게 10픽셀 여유를 두고 박스 설정
            x, y, w, h = best_white_box
            return (max(0, x-10), max(0, y-10), w+20, h+20)
            
        return None

    def solve_captcha(self):
        """
        거탐 팝업 시 호출됩니다.
        CSRT Tracker를 이용하여 투명해지는 도형을 물고 늘어집니다.
        """
        print("[거탐 AI] 도형 탐색 시작 (CSRT Tracking 준비 중...)")
        
        # 1. 초기 도형 찾기 (선명할 때)
        search_start = time.time()
        initial_bbox = None
        
        while time.time() - search_start < 10.0:
            screenshot = np.array(self.sct.grab(self.monitor))
            frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            bbox = self.find_initial_target(frame_bgr)
            if bbox is not None:
                initial_bbox = bbox
                print(f"[거탐 AI] 초기 흰색 바탕도형 발견! -> 위치: {initial_bbox}")
                break
                
            time.sleep(0.1)
            
        if initial_bbox is None:
            print("[거탐 AI] 10초 동안 명확한 도형을 찾지 못했습니다. 수동으로 마우스를 올려주세요!")
            # 사용자가 수동으로 마우스를 올렸다고 가정하고 현재 마우스 위치 주변을 강제로 Lock-on 할 수도 있습니다.
            # 하지만 여기서는 안전하게 실패 처리.
            return False
            
        # 2. 찾은 영역으로 CSRT Tracker 초기화
        tracker = cv2.TrackerCSRT_create()
        screenshot = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        tracker.init(frame, initial_bbox)
        print("[거탐 AI] CSRT Tracker Lock-on 완료! 추적을 시작합니다.")
        
        # 3. 투명해지는 도형 실시간 추적 및 마우스 이동
        track_start = time.time()
        while time.time() - track_start < 25.0: # 거탐 지속시간 동안 유지
            screenshot = np.array(self.sct.grab(self.monitor))
            frame_original = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(frame_original, cv2.COLOR_BGR2HSV)
            
            # 마우스 커서의 핑크색 '테두리'만 찾아내서 얇게 지움 (투명 타겟 중심부 보존)
            pink_mask = cv2.inRange(hsv, np.array([130, 30, 50]), np.array([175, 255, 255]))
            
            if cv2.countNonZero(pink_mask) > 0:
                kernel = np.ones((3,3), np.uint8)
                dilated_mask = cv2.dilate(pink_mask, kernel, iterations=1)
                dilated_mask[:, 1280:] = 0
                tracker_frame = cv2.inpaint(frame_original, dilated_mask, 3, cv2.INPAINT_TELEA)
            else:
                tracker_frame = frame_original.copy()
            
            success, bbox = tracker.update(tracker_frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 마우스를 추적된 중심 좌표로 이동 (1초에 20번씩 부드럽게 쫓아감)
                move_mouse(center_x, center_y)
                print(f"[거탐 AI] Tracking... ({center_x}, {center_y})")
            else:
                print("[거탐 AI] 추적 대상을 잃어버렸습니다!")
                break
                
            time.sleep(0.05) # 20 FPS
            
        print("[거탐 AI] 거탐 추적 사이클 종료.")
        return True

if __name__ == "__main__":
    solver = CaptchaSolver()
    solver.solve_captcha()
