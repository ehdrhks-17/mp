import sys
import os
import glob
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QLabel, QComboBox, QHBoxLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# bot_logic과 map_editor는 나중에 통합 연결됩니다.
# from bot_logic import BotThread
import subprocess

class MainGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.bot_thread = None
        
        # 디렉토리 체크
        os.makedirs("maps", exist_ok=True)
        
        self.initUI()
        self.refresh_lists()

    def initUI(self):
        # 윈도우 속성 설정 (항상 위)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        layout = QVBoxLayout()
        
        title = QLabel("🍁 메이플 봇 제어반")
        title.setFont(QFont("Malgun Gothic", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #4CAF50;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 1. 맵 선택기
        map_layout = QVBoxLayout()
        map_label = QLabel("🗺️ 사냥터 맵 파일 선택")
        map_label.setFont(QFont("Malgun Gothic", 14, QFont.Weight.Bold))
        self.map_combo = QComboBox()
        self.map_combo.setStyleSheet("padding: 10px; font-size: 16px; background-color: #444;")
        map_layout.addWidget(map_label)
        map_layout.addWidget(self.map_combo)
        layout.addLayout(map_layout)
        
        # 2. 몬스터 인공지능 선택기
        ai_layout = QVBoxLayout()
        ai_label = QLabel("🧠 몬스터 인공지능(YOLO) 선택")
        ai_label.setFont(QFont("Malgun Gothic", 14, QFont.Weight.Bold))
        self.ai_combo = QComboBox()
        self.ai_combo.setStyleSheet("padding: 10px; font-size: 16px; background-color: #444;")
        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.ai_combo)
        layout.addLayout(ai_layout)
        
        # 버튼 영역
        btn_layout = QHBoxLayout()
        
        self.editor_btn = QPushButton('🛠️ 맵 에디터 열기')
        self.editor_btn.setMinimumHeight(60)
        self.editor_btn.setStyleSheet("background-color: #2196F3; font-size: 18px; font-weight: bold;")
        self.editor_btn.clicked.connect(self.open_editor)
        
        self.refresh_btn = QPushButton('🔄 새로고침')
        self.refresh_btn.setMinimumHeight(60)
        self.refresh_btn.setStyleSheet("background-color: #757575; font-size: 18px; font-weight: bold;")
        self.refresh_btn.clicked.connect(self.refresh_lists)
        
        btn_layout.addWidget(self.editor_btn)
        btn_layout.addWidget(self.refresh_btn)
        layout.addLayout(btn_layout)
        
        # 거대한 시작 버튼
        self.toggle_btn = QPushButton('▶ 사냥 시작')
        self.toggle_btn.setMinimumHeight(120)
        self.toggle_btn.setFont(QFont("Malgun Gothic", 28, QFont.Weight.Bold))
        self.toggle_btn.setStyleSheet("background-color: #4CAF50; border: 3px solid #388E3C; border-radius: 15px;")
        self.toggle_btn.clicked.connect(self.toggle_bot)
        layout.addWidget(self.toggle_btn)

        self.setLayout(layout)
        self.setGeometry(100, 100, 450, 550)
        self.setWindowTitle('Maple YOLO Bot - Main GUI')

    def refresh_lists(self):
        # 맵 파일 불러오기 (.json)
        self.map_combo.clear()
        map_files = glob.glob(os.path.join("maps", "*.json"))
        if map_files:
            for f in map_files:
                self.map_combo.addItem(os.path.basename(f))
        else:
            self.map_combo.addItem("저장된 맵이 없습니다.")
            
        # 가중치 파일 불러오기 (.pt)
        self.ai_combo.clear()
        # 현재 폴더의 .pt 파일들
        pt_files = glob.glob("*.pt")
        if pt_files:
            for f in pt_files:
                self.ai_combo.addItem(os.path.basename(f))
        else:
            self.ai_combo.addItem("기본 YOLO 모델 (yolov8n.pt)")

    def open_editor(self):
        # 맵 에디터 스크립트 실행 (별도 창으로 띄움)
        subprocess.Popen([sys.executable, "map_editor.py"])

    def toggle_bot(self):
        # TODO: bot_logic과 연결하여 선택된 맵과 가중치를 전달하고 스레드를 시작하는 로직 구현
        if self.toggle_btn.text() == '▶ 사냥 시작':
            self.toggle_btn.setText("⏹ 사냥 정지")
            self.toggle_btn.setStyleSheet("background-color: #f44336; border: 3px solid #c62828; border-radius: 15px;")
            print(f"[{self.map_combo.currentText()}] 맵과 [{self.ai_combo.currentText()}] AI로 사냥을 시작합니다.")
        else:
            self.toggle_btn.setText("▶ 사냥 시작")
            self.toggle_btn.setStyleSheet("background-color: #4CAF50; border: 3px solid #388E3C; border-radius: 15px;")
            print("사냥을 정지합니다.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainGUI()
    ex.show()
    sys.exit(app.exec())
