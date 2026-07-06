import cv2
import numpy as np
import mss
import time
from controller import move_mouse

class CaptchaSolver:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.is_tracking = False

    def detect_captcha(self):
        """
        화면에서 거탐(투명 도형 찾기) UI가 존재하는지 감지합니다.
        (핑크색 마우스 포인터의 존재 여부로 판단)
        """
        img = np.array(self.sct.grab(self.monitor))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA, cv2.BGR)
        
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 100, 150])
        upper_pink = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 500:
            return None, None
            
        # 핑크색 포인터의 Bounding Box 리턴
        return cv2.boundingRect(largest_contour), img_bgr

    def check_and_solve(self):
        """
        메인 봇 루프에서 호출되며, 거탐이 감지되면 풀릴 때까지 제어권을 가져옵니다.
        """
        bbox, img_bgr = self.detect_captcha()
        if bbox is None:
            return False # 거탐 아님
            
        print("🚨 [경고] 투명 도형 거탐 감지됨! 마우스 트래킹을 시작합니다.")
        self.is_tracking = True
        
        # 첫 감지 시 타겟 도형(힌트) 템플릿 확보
        x, y, w, h = bbox
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        target_roi = img_gray[y:y+h, x:x+w]
        
        # 엣지 검출을 통해 특징을 잡아냄
        target_blur = cv2.GaussianBlur(target_roi, (5, 5), 0)
        edges_target = cv2.Canny(target_blur, 50, 150)

        # 거탐이 사라질 때까지 트래킹 루프 유지
        while self.is_tracking:
            current_bbox, current_bgr = self.detect_captcha()
            if current_bbox is None:
                # 거탐이 사라졌음 (성공)
                print("✅ 거탐이 해제되어 화면에서 사라졌습니다!")
                self.is_tracking = False
                break
                
            current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
            
            # 노이즈가 있는 전체 캔버스 영역 추정 (핑크색 포인터 위쪽 넓은 영역)
            px, py, pw, ph = current_bbox
            bg_y1 = max(0, py - 300)
            bg_y2 = py - 10
            bg_x1 = max(0, px - 200)
            bg_x2 = min(self.monitor['width'], px + 200 + pw)
            
            bg_roi = current_gray[bg_y1:bg_y2, bg_x1:bg_x2]
            
            if bg_roi.size > 0:
                bg_blur = cv2.GaussianBlur(bg_roi, (5, 5), 0)
                edges_bg = cv2.Canny(bg_blur, 50, 150)
                
                # 템플릿 매칭으로 투명 도형 현재 위치 찾기
                res = cv2.matchTemplate(edges_bg, edges_target, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # 시각화용 디버그 이미지 생성 (영상처럼 Answer 창 표시)
                debug_img = cv2.cvtColor(edges_bg, cv2.COLOR_GRAY2BGR)
                
                if max_val >= 0.2: # 에지 매칭이므로 비교적 낮게 설정
                    match_x = bg_x1 + max_loc[0] + (pw // 2)
                    match_y = bg_y1 + max_loc[1] + (ph // 2)
                    
                    # 마우스를 투명 도형 위치로 따라가기 (Follow)
                    move_mouse(match_x, match_y)
                    
                    # 디버그 윈도우에 빨간색 사각형 그리기
                    cv2.rectangle(debug_img, max_loc, (max_loc[0] + pw, max_loc[1] + ph), (0, 0, 255), 2)
                
                # AI Vision Debug 창 띄우기
                cv2.imshow("Answer (crop) - AI Vision Debug", debug_img)
                cv2.waitKey(1)
            
            # 0.05초 간격으로 계속 추적
            time.sleep(0.05)
            
        # 루프 종료 시 창 닫기
        cv2.destroyAllWindows()
        return True

if __name__ == "__main__":
    solver = CaptchaSolver()
    print("거탐 트래킹 테스트 중...")
    while True:
        solver.check_and_solve()
        time.sleep(1)
