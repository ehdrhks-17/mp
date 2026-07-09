import cv2
import numpy as np
import os

def extract_and_tile():
    # 1. 원본 프레임 로드
    img = cv2.imread('scratch/user_vid/frame_0.png')
    if img is None:
        print("프레임을 찾을 수 없습니다.")
        return
        
    # 2. 깨끗한 바위 패턴 영역 하드코딩 크롭 (별이 안 겹치는 구석 부분)
    # 전체 이미지 (640x280). x=480~540, y=200~260 영역이 비교적 깨끗함
    patch_size = 60
    patch = img[200:260, 480:540]
    
    # 패치를 경계선 없이 부드럽게 이어붙이기 위해 상하좌우를 블렌딩하거나 
    # OpenCV seamlessClone 등을 쓸 수 있지만, 타일 텍스처이므로 일단 그대로 이어붙입니다.
    # 경계선을 줄이기 위해 미러링(거울 반사) 타일링 기법을 사용합니다.
    
    h_patch, w_patch = patch.shape[:2]
    
    # 미러링 타일 블록 만들기 (상하좌우 대칭으로 2x2 사이즈 만들기 -> 경계선 완벽 제거)
    patch_flip_lr = cv2.flip(patch, 1)
    patch_top = np.hstack((patch, patch_flip_lr))
    patch_bottom = cv2.flip(patch_top, 0)
    seamless_tile = np.vstack((patch_top, patch_bottom))
    
    h_tile, w_tile = seamless_tile.shape[:2]
    
    # 3. 800x600 크기로 꽉 차게 타일링 (복제)
    target_w, target_h = 800, 600
    num_x = (target_w // w_tile) + 1
    num_y = (target_h // h_tile) + 1
    
    bg = np.tile(seamless_tile, (num_y, num_x, 1))
    
    # 정확한 크기로 자르기
    bg = bg[:target_h, :target_w]
    
    # 4. 저장
    cv2.imwrite('bg.png', bg)
    print("bg.png 배경 이미지 생성 완료!")

if __name__ == "__main__":
    extract_and_tile()
