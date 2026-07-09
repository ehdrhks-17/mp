import cv2
import numpy as np
import mss
import mouse
import time

class CaptchaSolver:
    def __init__(self, debug_callback=None):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # 전 화면 캡처용 (나중에 ROI 지정 가능)
        self.debug_callback = debug_callback
        
        self.state = "SEARCHING"
        self.tracker = None
        self.kalman = None
        self.last_box_size = (60, 60)
        
    def _preprocess(self, frame_bgr):
        # 1. 흑백 변환
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. 강한 가우시안 블러 (내부의 자잘한 바위 질감을 뭉개버려서 엣지에서 제외시킴)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # 3. Canny 윤곽선 추출 (오직 뾰족한 '별의 뼈대'만 하얀 선으로 남김)
        # 임계값을 적절히 주어 뚜렷한 경계선만 취함
        edges = cv2.Canny(blurred, 30, 100)
        
        # 4. Dilation (선 팽창) - 얇은 뼈대를 두껍게 칠해서 트래커가 꽉 물기 좋게 만듦
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        return dilated

    def _init_kalman(self, x, y):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        # 속도 모델 (관성 유지)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],
                                                [0,1,0,0],
                                                [0,0,5,0],
                                                [0,0,0,5]], np.float32) * 0.03
        
        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    def find_initial_target(self, frame_bgr):
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
        
        preprocessed = self._preprocess(frame_bgr)
        # CSRT 트래커는 3채널을 요구함
        preprocessed_3c = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        
        if self.state == "SEARCHING":
            rect = self.find_initial_target(frame_bgr)
            if rect:
                x, y, w, h = rect
                # 락온! 트래커 시작
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(preprocessed_3c, (x, y, w, h))
                
                cx, cy = x + w/2, y + h/2
                self._init_kalman(cx, cy)
                self.last_box_size = (w, h)
                
                self.state = "TRACKING"
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, "LOCKED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif self.state == "TRACKING":
            # 1. 칼만 필터 예측 (다음 프레임에 있어야 할 관성 위치)
            pred = self.kalman.predict()
            pred_x, pred_y = float(pred[0][0]), float(pred[1][0])
            
            # 2. CSRT 트래커 실제 측정
            success, box = self.tracker.update(preprocessed_3c)
            
            if success:
                tx, ty, tw, th = [int(v) for v in box]
                self.last_box_size = (tw, th)
                meas_cx = tx + tw/2
                meas_cy = ty + th/2
                
                # 예측된 위치와 실제 트래커가 찾은 위치의 거리 오차
                dist = np.sqrt((meas_cx - pred_x)**2 + (meas_cy - pred_y)**2)
                
                # 가짜 별과 교차(Occlusion) 시 트래커가 순간적으로 튀는 현상 방어!
                if dist > tw * 2.5: # 거리를 2.5배로 늘려 관대하게 허용
                    # 쉴드 발동: 트래커가 너무 멀리 튀었다면 관성 예측값을 강제로 사용
                    cv2.putText(debug_img, "OCCLUSION SHIELD!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    final_cx, final_cy = pred_x, pred_y
                    
                    # 트래커를 파괴하지 않고, 칼만 필터의 예측값만으로 관성을 유지함!
                    # (중간에 투명해진 상태로 트래커를 재시작하면 배경을 학습해버려서 추적이 영원히 망가짐)
                else:
                    # 정상 추적 중: 칼만 필터에 측정값 먹여서 관성 궤도 교정
                    measurement = np.array([[np.float32(meas_cx)], [np.float32(meas_cy)]])
                    self.kalman.correct(measurement)
                    final_cx, final_cy = meas_cx, meas_cy
                    
                # 시각화 박스 그리기
                tw, th = self.last_box_size
                cv2.rectangle(debug_img, (int(final_cx-tw/2), int(final_cy-th/2)), 
                              (int(final_cx+tw/2), int(final_cy+th/2)), (255, 255, 0), 2)
                cv2.circle(debug_img, (int(pred_x), int(pred_y)), 5, (0, 0, 255), -1) # 빨간점: 칼만 예측
                cv2.circle(debug_img, (int(meas_cx), int(meas_cy)), 3, (0, 255, 0), -1) # 초록점: 트래커 측정
                
                target_center_screen = (int(final_cx) + screen_offset_x, int(final_cy) + screen_offset_y)
                
            else:
                self.state = "SEARCHING"
                cv2.putText(debug_img, "LOST! SEARCHING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if target_center_screen:
            # 즉각적으로 마우스 이동
            mouse.move(target_center_screen[0], target_center_screen[1], absolute=True, duration=0)
            
        if self.debug_callback:
            self.debug_callback(debug_img, preprocessed_3c)
            
        return debug_img
