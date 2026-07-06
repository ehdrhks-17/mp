import cv2
import numpy as np
import mss
import time
from controller import tap_key, click_mouse, move_mouse

class CaptchaSolver:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        
        # 기본 제공 템플릿 모양 3가지 (별, 원, 세모) - 처음 선명할 때 잡기 위함
        self.templates = {
            'star': self._create_star_template(),
            'circle': self._create_circle_template(),
            'triangle': self._create_triangle_template()
        }

    def _create_star_template(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        pts = np.array([[25,5], [31,18], [45,18], [34,26], [38,40], [25,31], [12,40], [16,26], [5,18], [19,18]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _create_circle_template(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(mask, (25, 25), 18, 255, -1)
        return mask

    def _create_triangle_template(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        pts = np.array([[25, 5], [5, 45], [45, 45]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def find_initial_target(self, frame_gray, frame_bgr):
        """
        초기에 도형이 투명해지기 전(또는 선명할 때) 모양을 찾아냅니다.
        """
        # 마우스 포인터(핑크색 원 내부의 흰색 도형)를 진짜 타겟으로 오인하지 않도록 마스킹
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 100, 150])
        upper_pink = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # 핑크색 커서 영역을 부드럽게 뭉개서(Blur) 추적기가 마우스를 형체로 인식하지 못하게 함
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if 100 < cv2.contourArea(c) < 5000:
                x, y, w, h = cv2.boundingRect(c)
                # 안전하게 영역 확장
                x, y = max(0, x-10), max(0, y-10)
                w, h = min(frame_gray.shape[1]-x, w+20), min(frame_gray.shape[0]-y, h+20)
                
                # BGR과 Gray 프레임 모두에서 마우스 포인터를 뭉개버림
                roi_gray = frame_gray[y:y+h, x:x+w]
                frame_gray[y:y+h, x:x+w] = cv2.medianBlur(roi_gray, 31)
                
                roi_bgr = frame_bgr[y:y+h, x:x+w]
                frame_bgr[y:y+h, x:x+w] = cv2.medianBlur(roi_bgr, 31)

        best_val = 0
        best_loc = None
        best_shape = None
        
        for name, tmpl in self.templates.items():
            res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_shape = name
                
        # 매칭률이 0.5 이상이면 찾은 것으로 간주 (배경 노이즈 고려)
        if best_val >= 0.5 and best_loc is not None:
            return (best_loc[0], best_loc[1], 50, 50), best_shape, best_val
        return None, None, best_val

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
            frame_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
            
            bbox, shape_name, conf = self.find_initial_target(frame_gray, frame_bgr)
            if bbox is not None:
                initial_bbox = bbox
                print(f"[거탐 AI] 초기 도형({shape_name}) 발견! 신뢰도: {conf:.2f} -> 위치: {initial_bbox}")
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
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # 실시간 추적 중에도 마우스 포인터를 화면에서 지워버려야 함 (안 그러면 마우스를 쫓아감)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pink_mask = cv2.inRange(hsv, np.array([140, 100, 150]), np.array([170, 255, 255]))
            contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if 100 < cv2.contourArea(c) < 5000:
                    x, y, w, h = cv2.boundingRect(c)
                    x, y = max(0, x-10), max(0, y-10)
                    w, h = min(frame.shape[1]-x, w+20), min(frame.shape[0]-y, h+20)
                    roi = frame[y:y+h, x:x+w]
                    frame[y:y+h, x:x+w] = cv2.medianBlur(roi, 31)
            
            success, bbox = tracker.update(frame)
            
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
