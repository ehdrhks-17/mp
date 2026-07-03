import cv2
import os
import glob

class DragLabeler:
    def __init__(self, dataset_dir="dataset"):
        self.img_dir = os.path.join(dataset_dir, "images", "train")
        self.label_dir = os.path.join(dataset_dir, "labels", "train")
        os.makedirs(self.label_dir, exist_ok=True)
        
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_box = None
        self.boxes = [] # [(x, y, w, h), ...]
        
        self.current_img_copy = None

    def draw_box(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_temp = self.current_img_copy.copy()
                cv2.rectangle(img_temp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Labeler (Drag & Drop)', img_temp)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            
            # 너무 작은 박스 무시
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                self.boxes.append((x1, y1, x2 - x1, y2 - y1))
                cv2.rectangle(self.current_img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('Labeler (Drag & Drop)', self.current_img_copy)

    def run(self):
        images = glob.glob(os.path.join(self.img_dir, "*.png"))
        images += glob.glob(os.path.join(self.img_dir, "*.jpg"))
        
        if not images:
            print("수집된 이미지가 없습니다.")
            return

        cv2.namedWindow('Labeler (Drag & Drop)')
        cv2.setMouseCallback('Labeler (Drag & Drop)', self.draw_box)
        
        print("--- 드래그 라벨링 시작 ---")
        print("드래그: 몬스터 박스 그리기")
        print("Space 또는 Enter: 다음 사진으로 (저장)")
        print("c: 그려진 박스 모두 취소")
        print("s: 허공 사진이라 몬스터 없음 (그냥 넘기기)")
        print("q: 종료")

        for img_path in images:
            # 해당 이미지의 라벨이 이미 존재하면 건너뜀
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, f"{base_name}.txt")
            if os.path.exists(label_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_h, img_w = img.shape[:2]
            self.current_img_copy = img.copy()
            self.boxes = []
            
            while True:
                cv2.imshow('Labeler (Drag & Drop)', self.current_img_copy)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '): # Space (Next & Save)
                    # 박스들을 YOLO 포맷으로 저장
                    if self.boxes:
                        with open(label_path, 'w') as f:
                            for (bx, by, bw, bh) in self.boxes:
                                # YOLO format: class x_center y_center width height (normalized 0~1)
                                cx = (bx + bw / 2) / img_w
                                cy = (by + bh / 2) / img_h
                                norm_w = bw / img_w
                                norm_h = bh / img_h
                                f.write(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                        print(f"[{base_name}] 저장 완료! ({len(self.boxes)}개)")
                    else:
                        print(f"[{base_name}] 빈 라벨로 저장됨.")
                        with open(label_path, 'w') as f: pass
                    break
                    
                elif key == ord('c'): # Clear boxes
                    self.current_img_copy = img.copy()
                    self.boxes = []
                    
                elif key == ord('s'): # Skip (Save as empty label)
                    print(f"[{base_name}] 허공 사진 패스")
                    with open(label_path, 'w') as f: pass
                    break
                    
                elif key == ord('q'): # Quit
                    cv2.destroyAllWindows()
                    return
                    
        cv2.destroyAllWindows()
        print("모든 라벨링이 완료되었습니다!")

if __name__ == "__main__":
    app = DragLabeler()
    app.run()
