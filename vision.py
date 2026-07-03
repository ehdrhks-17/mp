import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time
import os

class VisionAgent:
    def __init__(self, yolo_model_path='best.pt'):
        self.sct = mss.mss()
        # 모니터 전체 해상도를 가져옵니다. 1번 모니터 기준.
        self.monitor = self.sct.monitors[1]
        
        # YOLO 모델 로드 (파일이 없으면 경고 출력 후 None 처리)
        self.model = None
        if os.path.exists(yolo_model_path):
            try:
                self.model = YOLO(yolo_model_path)
                print(f"YOLO 모델({yolo_model_path}) 로드 성공.")
            except Exception as e:
                print(f"YOLO 모델 로드 실패: {e}")
        else:
            print(f"경고: {yolo_model_path} 파일이 없습니다. YOLO 인식 없이 미니맵 타 유저 감지만 작동합니다.")

    def get_screen(self, region=None):
        """지정된 영역(없으면 전체)의 화면을 캡처하여 OpenCV BGR 이미지로 반환"""
        cap_region = region if region else self.monitor
        img = np.array(self.sct.grab(cap_region))
        # mss는 BGRA로 캡처하므로 BGR로 변환
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect_red_dots(self, minimap_region=None):
        """
        미니맵 영역에서 빨간색 점(타 유저)을 감지합니다.
        minimap_region: {'top': y, 'left': x, 'width': w, 'height': h} 형태의 딕셔너리
        기본값은 좌측 상단 대략적인 400x300 영역으로 설정합니다.
        """
        if not minimap_region:
            minimap_region = {'top': 0, 'left': 0, 'width': 400, 'height': 300}
            
        img = self.get_screen(minimap_region)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 메이플스토리 미니맵의 빨간점 색상 범위 (HSV)
        # 빨간색은 Hue 값이 0 근처와 180 근처 양쪽에 걸쳐 있습니다.
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 150])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # 빨간색 픽셀의 개수가 일정량 이상이면 타 유저가 있는 것으로 간주
        red_pixels = cv2.countNonZero(mask)
        # 픽셀 임계값은 미니맵 크기에 따라 조절 필요 (예: 5 픽셀 이상)
        if red_pixels > 5:
            return True
        return False

    def find_monsters(self, region=None):
        """
        YOLO 모델을 사용하여 화면에서 몬스터의 Bounding Box를 반환합니다.
        반환값: [(x_center, y_center, width, height, confidence), ...]
        """
        if self.model is None:
            return []
            
        img = self.get_screen(region)
        results = self.model(img, verbose=False)
        
        monsters = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # conf threshold 설정 (예: 0.5 이상)
                if box.conf[0] > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w // 2
                    cy = y1 + h // 2
                    monsters.append((cx, cy, w, h, box.conf[0].item()))
    def find_my_character(self, minimap_region=None):
        """
        미니맵에서 내 캐릭터(노란색 점)의 중심 (x, y) 좌표를 찾습니다.
        """
        if not minimap_region:
            minimap_region = {'top': 0, 'left': 0, 'width': 400, 'height': 300}
            
        img = self.get_screen(minimap_region)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 메이플스토리 미니맵 노란점 색상 범위
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 노란색 픽셀들의 무게중심(좌표 평균) 계산
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))
            return (cx, cy)
        return None

if __name__ == "__main__":
    # Test block
    v = VisionAgent()
    print("Testing Red Dot Detection...")
    has_red = v.detect_red_dots()
    print(f"Red dot detected: {has_red}")
