import sys
import os
import json
import mss
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QPushButton, QHBoxLayout, QInputDialog, QMessageBox, QGridLayout)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QRect

class MapEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.map_dir = "maps"
        os.makedirs(self.map_dir, exist_ok=True)
        
        self.nodes = [] # List of dicts: {'x': int, 'y': int, 'action': str}
        self.box_size = 40 # 미니맵 상의 박스 크기
        
        self.capture_minimap()
        self.initUI()

    def capture_minimap(self):
        # 화면 좌측 상단 미니맵 영역 캡처 (대략 400x300)
        sct = mss.mss()
        monitor = sct.monitors[1]
        region = {'top': 0, 'left': 0, 'width': 400, 'height': 300}
        img_bgra = np.array(sct.grab(region))
        self.bg_img = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
        
        # Convert to QPixmap
        h, w, ch = self.bg_img.shape
        bytes_per_line = ch * w
        q_img = QImage(self.bg_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_img)

    def initUI(self):
        self.setWindowTitle("미니맵 액션 노드 에디터 (드래그 X)")
        
        layout = QVBoxLayout()
        
        info = QLabel("미니맵 캡처 완료! 원하는 곳을 클릭하여 행동 박스를 설치하세요.")
        info.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        layout.addWidget(info)
        
        # 이미지 레이블
        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.original_pixmap)
        self.image_label.mousePressEvent = self.on_image_click
        layout.addWidget(self.image_label)
        
        # 액션 선택 팝업 패널 (초기엔 숨김)
        self.action_panel = QWidget()
        action_layout = QGridLayout()
        self.action_panel.setLayout(action_layout)
        
        actions = [
            ("⬅ 좌로 걷기", "walk_left"), ("우로 걷기 ➡", "walk_right"),
            ("↖ 좌로 점프", "jump_left"), ("우로 점프 ↗", "jump_right"),
            ("⬆ 줄타기", "rope_up"), ("⬇ 밑점프", "jump_down"),
            ("❌ 취소", "cancel")
        ]
        
        self.action_btns = {}
        row, col = 0, 0
        for text, code in actions:
            btn = QPushButton(text)
            btn.setMinimumHeight(50)
            btn.setStyleSheet("font-size: 16px; font-weight: bold;")
            btn.clicked.connect(lambda checked, c=code: self.add_node(c))
            action_layout.addWidget(btn, row, col)
            self.action_btns[code] = btn
            col += 1
            if col > 1:
                col = 0
                row += 1
                
        self.action_panel.hide()
        layout.addWidget(self.action_panel)
        
        # 하단 컨트롤
        control_layout = QHBoxLayout()
        save_btn = QPushButton("💾 맵 파일로 저장하기")
        save_btn.setMinimumHeight(60)
        save_btn.setStyleSheet("background-color: #FF9800; color: white; font-size: 18px; font-weight: bold;")
        save_btn.clicked.connect(self.save_map)
        
        clear_btn = QPushButton("🗑️ 모두 지우기")
        clear_btn.setMinimumHeight(60)
        clear_btn.clicked.connect(self.clear_nodes)
        
        control_layout.addWidget(clear_btn)
        control_layout.addWidget(save_btn)
        layout.addLayout(control_layout)
        
        self.setLayout(layout)
        self.temp_x = 0
        self.temp_y = 0

    def on_image_click(self, event):
        self.temp_x = event.pos().x()
        self.temp_y = event.pos().y()
        # 팝업 띄우기
        self.action_panel.show()
        
    def add_node(self, action_code):
        self.action_panel.hide()
        if action_code == "cancel":
            return
            
        self.nodes.append({
            'x': self.temp_x,
            'y': self.temp_y,
            'action': action_code
        })
        self.update_display()

    def update_display(self):
        # Draw on a copy of the original pixmap
        pixmap = self.original_pixmap.copy()
        painter = QPainter(pixmap)
        
        for node in self.nodes:
            x, y = node['x'], node['y']
            action = node['action']
            
            # 행동별 색상 지정
            if "left" in action: color = QColor(33, 150, 243, 100) # 파랑
            elif "right" in action: color = QColor(244, 67, 54, 100) # 빨강
            elif "rope" in action: color = QColor(76, 175, 80, 100) # 초록
            else: color = QColor(255, 152, 0, 100) # 주황
            
            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            # 중앙 기준 박스 그리기
            rect = QRect(x - self.box_size//2, y - self.box_size//2, self.box_size, self.box_size)
            painter.drawRect(rect)
            
            # 텍스트 그리기
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, action.split('_')[0])
            
        painter.end()
        self.image_label.setPixmap(pixmap)

    def clear_nodes(self):
        self.nodes = []
        self.update_display()

    def save_map(self):
        if not self.nodes:
            QMessageBox.warning(self, "경고", "배치된 노드가 없습니다!")
            return
            
        text, ok = QInputDialog.getText(self, '맵 저장', '저장할 사냥터 이름 (예: 개미굴_1층):')
        if ok and text:
            filename = os.path.join(self.map_dir, f"{text}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.nodes, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "성공", f"[{text}.json] 맵이 저장되었습니다!")
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MapEditor()
    ex.show()
    sys.exit(app.exec())
