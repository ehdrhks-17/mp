import sys
import os
import glob
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QRect

class OneClickLabeler(QWidget):
    def __init__(self):
        super().__init__()
        # 경로 설정
        self.img_dir = os.path.join("dataset", "images", "train")
        self.label_dir = os.path.join("dataset", "labels", "train")
        os.makedirs(self.label_dir, exist_ok=True)
        
        # 이미지 파일 목록 불러오기 (.jpg)
        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.current_index = 0
        
        # 몬스터 네모 박스 크기 (고정)
        # 메이플 몬스터는 대략 가로 100~150픽셀 정도입니다.
        self.box_size = 120 
        
        self.initUI()
        self.load_image()

    def initUI(self):
        self.setWindowTitle("YOLO 원클릭 라벨링 툴 (드래그 X)")
        
        layout = QVBoxLayout()
        
        # 안내 문구
        self.info_label = QLabel("몬스터의 정중앙을 눈으로 보고 딱 한 번만 클릭(깜빡임)하세요!\n자동으로 박스가 쳐지고 다음 사진으로 넘어갑니다.", self)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        layout.addWidget(self.info_label)
        
        # 진행 상황
        self.progress_label = QLabel("", self)
        self.progress_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.progress_label)
        
        # 이미지 표시 영역
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.mousePressEvent = self.on_image_click
        layout.addWidget(self.image_label)
        
        # 컨트롤 버튼
        btn_layout = QHBoxLayout()
        self.skip_btn = QPushButton("건너뛰기 (몬스터 없음)", self)
        self.skip_btn.setMinimumHeight(50)
        self.skip_btn.clicked.connect(self.next_image)
        
        self.prev_btn = QPushButton("이전 사진", self)
        self.prev_btn.setMinimumHeight(50)
        self.prev_btn.clicked.connect(self.prev_image)
        
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.skip_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        # 윈도우 크기를 화면 크기에 맞게 크게
        self.setGeometry(100, 100, 1280, 720)

    def load_image(self):
        if not self.image_files:
            QMessageBox.information(self, "알림", "저장된 사진이 없습니다! data_collector.py를 먼저 실행하세요.")
            sys.exit()
            
        if self.current_index >= len(self.image_files):
            QMessageBox.information(self, "완료", "모든 사진의 라벨링이 끝났습니다! 이제 train.py를 실행하세요.")
            sys.exit()
            
        self.current_img_path = self.image_files[self.current_index]
        self.pixmap = QPixmap(self.current_img_path)
        
        # 레이블 크기에 맞게 이미지 축소 (비율 유지)
        scaled_pixmap = self.pixmap.scaled(
            1280, 720, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
        # 원본과 화면에 표시된 이미지 간의 스케일 비율 계산
        self.scale_factor = self.pixmap.width() / scaled_pixmap.width()
        
        self.progress_label.setText(f"진행 상황: {self.current_index + 1} / {len(self.image_files)}")

    def on_image_click(self, event):
        # 1. 클릭 좌표 얻기 (화면에 보이는 이미지 기준)
        x = event.pos().x()
        y = event.pos().y()
        
        # 2. 이미지 위젯 내에서 실제 이미지가 표시되는 영역(여백 제외) 계산
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = self.image_label.pixmap().width()
        pixmap_height = self.image_label.pixmap().height()
        
        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2
        
        # 여백을 클릭했다면 무시
        if x < offset_x or x > offset_x + pixmap_width or y < offset_y or y > offset_y + pixmap_height:
            return
            
        # 3. 원본 이미지 해상도 기준의 좌표로 변환
        real_x = (x - offset_x) * self.scale_factor
        real_y = (y - offset_y) * self.scale_factor
        
        # 4. YOLO 정답 파일(txt) 생성 및 저장
        self.save_yolo_label(real_x, real_y)
        
        # 5. 시각적 피드백 (박스 그려서 잠깐 보여주기)
        # 이 부분은 생략하고 바로 다음 사진으로 넘어가서 피로도를 줄입니다.
        self.next_image()

    def save_yolo_label(self, center_x, center_y):
        # 원본 이미지 가로, 세로
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        
        # 정규화된 값 계산 (0.0 ~ 1.0)
        norm_x = center_x / img_w
        norm_y = center_y / img_h
        norm_w = self.box_size / img_w
        norm_h = self.box_size / img_h
        
        # 텍스트 파일명 생성 (이미지 파일명과 동일하되 확장자만 .txt)
        base_name = os.path.basename(self.current_img_path)
        txt_name = os.path.splitext(base_name)[0] + ".txt"
        txt_path = os.path.join(self.label_dir, txt_name)
        
        # YOLO 형식: class_id x_center y_center width height
        # class_id는 0 (몬스터 1종류만 있다고 가정)
        with open(txt_path, 'w') as f:
            f.write(f"0 {norm_x} {norm_y} {norm_w} {norm_h}\n")

    def next_image(self):
        self.current_index += 1
        self.load_image()
        
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = OneClickLabeler()
    ex.show()
    sys.exit(app.exec())
