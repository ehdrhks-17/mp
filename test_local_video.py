import cv2
import time
from captcha_solver import CaptchaSolver

def test_video_tracking():
    video_path = r'C:\Users\DK\Downloads\녹화_2026_07_09_00_28_18_662_trim.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 화면에 띄울 이미지를 리턴받기 위한 콜백 함수
    def debug_callback(debug_img, prep_img):
        # 영상이 너무 클 수 있으므로 적절히 리사이즈
        h, w = debug_img.shape[:2]
        scale = 800 / w if w > 800 else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        
        d_img = cv2.resize(debug_img, (new_w, new_h))
        p_img = cv2.resize(prep_img, (new_w, new_h))
        
        cv2.imshow("Main Tracking View", d_img)
        cv2.imshow("Edge Preprocessing View", p_img)

    solver = CaptchaSolver(debug_callback=debug_callback)
    
    print("Starting video test... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        # 실제 환경과 동일하게 봇에 프레임 먹이기 (오프셋 0)
        # 마우스 이동 함수(mouse.move)는 봇 내부에 있으나 오프라인 디버그 중엔 끄거나 시각화만 봄.
        # 영상 테스트를 위해 solver 코드의 mouse.move를 잠시 비활성화하는게 좋지만,
        # 일단 봇 로직이 트래커 박스를 잘 그리는지만 확인합니다.
        
        solver.process_frame(frame, 0, 0)
        
        # 30fps 속도에 맞추거나, 스페이스바로 한 프레임씩 넘기게 대기
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_video_tracking()
