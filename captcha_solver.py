import cv2
import numpy as np
import mss
import time
import os
from controller import tap_key, click_mouse

class CaptchaSolver:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        
        # 투명 도형 찾기 UI의 타이틀 바나 특징점 이미지가 있다면 로드합니다.
        # 현재는 화면 중앙 부근에 거탐이 뜬다고 가정하고 영역을 설정합니다.
        self.is_active = False

    def check_and_solve(self):
        """
        화면에 거탐(투명 도형 찾기)이 떴는지 확인하고, 떴다면 자동으로 풉니다.
        """
        img = np.array(self.sct.grab(self.monitor))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA, cv2.BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # TODO: 실제 게임 화면에서 '투명 도형 찾기' 글씨나 핑크색 원을 템플릿 매칭으로 찾아야 함.
        # 여기서는 영상(유튜브)에서 분석한 알고리즘 기법을 적용한 뼈대를 구현합니다.
        
        # 1. 핑크색 원(제시된 타겟 도형) 위치 찾기 (예시: 특정 핑크색 범위 탐색)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 100, 150])
        upper_pink = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False # 거탐이 없음
            
        # 가장 큰 핑크색 원 찾기
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 500:
            return False # 너무 작으면 오인식
            
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        print("🚨 [경고] 투명 도형 거탐 감지됨! 자동 해제를 시도합니다.")
        
        # 2. 타겟 도형 크롭
        target_roi = img_gray[y:y+h, x:x+w]
        
        # 3. 거탐 배경 영역 크롭 (보통 핑크색 원 바로 위나 주변에 큰 노이즈 배경이 있음)
        # 영상 기준: 핑크색 원 위쪽 넓은 영역
        bg_y1 = max(0, y - 300)
        bg_y2 = y - 10
        bg_x1 = max(0, x - 200)
        bg_x2 = min(self.monitor['width'], x + 200 + w)
        
        bg_roi = img_gray[bg_y1:bg_y2, bg_x1:bg_x2]
        
        if bg_roi.size == 0 or target_roi.size == 0:
            return False
            
        # 4. 핵심 알고리즘: Canny Edge Detection을 사용해 투명 도형의 윤곽선을 도출
        # 노이즈가 많으므로 약간의 블러 처리 후 에지를 따냅니다.
        target_blur = cv2.GaussianBlur(target_roi, (5, 5), 0)
        bg_blur = cv2.GaussianBlur(bg_roi, (5, 5), 0)
        
        edges_target = cv2.Canny(target_blur, 50, 150)
        edges_bg = cv2.Canny(bg_blur, 50, 150)
        
        # 5. Template Matching (에지 이미지끼리 매칭하여 투명 도형 위치 탐색)
        res = cv2.matchTemplate(edges_bg, edges_target, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        threshold = 0.3 # 에지 매칭이므로 임계값을 다소 낮게 설정
        
        if max_val >= threshold:
            match_x = bg_x1 + max_loc[0] + (w // 2)
            match_y = bg_y1 + max_loc[1] + (h // 2)
            
            print(f"✅ 투명 도형 발견! 확률: {max_val*100:.1f}%. 좌표: ({match_x}, {match_y}) 클릭합니다.")
            click_mouse(match_x, match_y)
            time.sleep(1)
            return True
        else:
            print(f"❌ 투명 도형을 찾지 못했습니다. (최대 확률: {max_val*100:.1f}%)")
            return False

if __name__ == "__main__":
    solver = CaptchaSolver()
    print("거탐 감지 대기 중...")
    while True:
        solver.check_and_solve()
        time.sleep(2)
