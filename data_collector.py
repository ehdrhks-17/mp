import time
import os
import cv2
import mss
import numpy as np

def collect_data():
    # 데이터셋 폴더 구조 생성
    base_dir = "dataset"
    img_dir = os.path.join(base_dir, "images", "train")
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"[{img_dir}] 폴더에 사진 저장을 시작합니다.")
    print("게임을 플레이하시거나 구경하세요. 2초마다 자동으로 사진을 찍습니다.")
    print("종료하려면 스크립트 창에서 Ctrl+C를 누르세요.")
    
    sct = mss.mss()
    # 1번 모니터 전체 화면 기준
    monitor = sct.monitors[1]
    
    count = 1
    # 기존 파일이 있다면 이어서 번호 매기기
    while os.path.exists(os.path.join(img_dir, f"maple_{count:04d}.jpg")):
        count += 1
        
    try:
        while True:
            # 2초 대기
            time.sleep(2)
            
            # 화면 캡처
            img_bgra = np.array(sct.grab(monitor))
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            
            # 파일 저장
            filename = os.path.join(img_dir, f"maple_{count:04d}.jpg")
            cv2.imwrite(filename, img_bgr)
            print(f"📸 찰칵! 저장됨: {filename}")
            
            count += 1
            
            # 100장이 넘어가면 알림 (권장 수량)
            if count == 100:
                print("\n🎉 100장의 사진이 모였습니다! 원하신다면 이제 라벨링을 시작해도 좋습니다.\n")
                
    except KeyboardInterrupt:
        print("\n데이터 수집이 종료되었습니다. 이제 one_click_labeler.py를 실행하세요!")

if __name__ == "__main__":
    collect_data()
