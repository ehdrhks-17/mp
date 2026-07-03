import sys
import os
import glob
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QLabel, QComboBox, QHBoxLayout, QLineEdit, QCheckBox, QSpinBox, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import subprocess

from bot_logic import BotThread
from vision import VisionAgent

class MainGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.bot_thread = None
        
        os.makedirs("maps", exist_ok=True)
        os.makedirs("mobs", exist_ok=True)
        self.initUI()
        self.refresh_lists()

    def initUI(self):
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        layout = QVBoxLayout()
        
        title = QLabel("🍁 MP 봇 제어반")
        title.setFont(QFont("Malgun Gothic", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #4CAF50;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 1. 맵 & 인공지능 선택
        h_layout1 = QHBoxLayout()
        
        map_layout = QVBoxLayout()
        map_layout.addWidget(QLabel("🗺️ 맵 파일"))
        self.map_combo = QComboBox()
        self.map_combo.setStyleSheet("padding: 5px; background: #444;")
        map_layout.addWidget(self.map_combo)
        
        ai_layout = QVBoxLayout()
        ai_layout.addWidget(QLabel("🧠 AI 모델"))
        self.ai_combo = QComboBox()
        self.ai_combo.setStyleSheet("padding: 5px; background: #444;")
        ai_layout.addWidget(self.ai_combo)
        
        h_layout1.addLayout(map_layout)
        h_layout1.addLayout(ai_layout)
        layout.addLayout(h_layout1)

        # 맵 에디터 버튼
        self.editor_btn = QPushButton('🛠️ 미니맵 노드 에디터 열기')
        self.editor_btn.setStyleSheet("background-color: #2196F3; padding: 10px; font-weight: bold;")
        self.editor_btn.clicked.connect(self.open_editor)
        layout.addWidget(self.editor_btn)

        # 2. 닉네임 추적 및 키 세팅
        title2 = QLabel("🎯 닉네임 캡처 & ⌨️ 키 세팅")
        title2.setStyleSheet("color: #FF9800; font-weight: bold; margin-top: 10px;")
        layout.addWidget(title2)
        
        self.capture_btn = QPushButton('📷 내 닉네임 이미지 따기 (저장)')
        self.capture_btn.setStyleSheet("background-color: #9C27B0; padding: 10px; font-weight: bold;")
        self.capture_btn.clicked.connect(self.capture_nickname)
        layout.addWidget(self.capture_btn)
        
        key_layout = QHBoxLayout()
        
        def make_key_input(label_text, default_val):
            vl = QVBoxLayout()
            vl.addWidget(QLabel(label_text))
            inp = QLineEdit(default_val)
            inp.setStyleSheet("background: #444; padding: 5px;")
            vl.addWidget(inp)
            return vl, inp
            
        vl1, self.key_attack = make_key_input("공격 키", "ctrl")
        vl2, self.key_jump = make_key_input("점프 키", "alt")
        vl3, self.key_teleport = make_key_input("텔포 키", "shift")
        
        key_layout.addLayout(vl1)
        key_layout.addLayout(vl2)
        key_layout.addLayout(vl3)
        layout.addLayout(key_layout)

        # 3. 전투 상세 설정
        title3 = QLabel("⚙️ 전투 상세 설정")
        title3.setStyleSheet("color: #FF9800; font-weight: bold; margin-top: 10px;")
        layout.addWidget(title3)
        
        range_layout = QHBoxLayout()
        
        def make_spinbox(label_text, default_val, max_val=2000):
            vl = QVBoxLayout()
            vl.addWidget(QLabel(label_text))
            sp = QSpinBox()
            sp.setRange(0, max_val)
            sp.setValue(default_val)
            sp.setStyleSheet("background: #444; padding: 5px;")
            vl.addWidget(sp)
            return vl, sp
            
        vl4, self.range_x = make_spinbox("인식 X (좌우)", 300)
        vl5, self.range_y = make_spinbox("인식 Y (상하)", 150)
        vl6, self.min_mobs = make_spinbox("최소 몹 수", 2, 20)
        
        range_layout.addLayout(vl4)
        range_layout.addLayout(vl5)
        range_layout.addLayout(vl6)
        layout.addLayout(range_layout)
        
        self.chk_jump_atk = QCheckBox("🦘 점프 공격 사용")
        self.chk_jump_atk.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.chk_jump_atk)
        
        self.chk_teleport = QCheckBox("🪄 이동 시 텔레포트 사용")
        self.chk_teleport.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.chk_teleport)

        # 시작 버튼
        self.toggle_btn = QPushButton('▶ 사냥 시작')
        self.toggle_btn.setMinimumHeight(100)
        self.toggle_btn.setFont(QFont("Malgun Gothic", 28, QFont.Weight.Bold))
        self.toggle_btn.setStyleSheet("background-color: #4CAF50; border: 3px solid #388E3C; border-radius: 15px; margin-top: 10px;")
        self.toggle_btn.clicked.connect(self.toggle_bot)
        layout.addWidget(self.toggle_btn)

        self.setLayout(layout)
        self.setGeometry(100, 100, 480, 750)
        self.setWindowTitle('MP YOLO Bot')

    def refresh_lists(self):
        self.map_combo.clear()
        map_files = glob.glob(os.path.join("maps", "*.json"))
        if map_files:
            for f in map_files:
                self.map_combo.addItem(os.path.basename(f))
        else:
            self.map_combo.addItem("저장된 맵이 없습니다.")
            
        self.ai_combo.clear()
        pt_files = glob.glob(os.path.join("mobs", "*.pt"))
        if pt_files:
            for f in pt_files:
                self.ai_combo.addItem(os.path.basename(f))
        else:
            self.ai_combo.addItem("기본 YOLO 모델 (yolov8n.pt)")

    def open_editor(self):
        subprocess.Popen([sys.executable, "map_editor.py"])
        
    def capture_nickname(self):
        vision = VisionAgent(yolo_model_path=None) # YOLO 필요없음
        if vision.capture_nickname():
            QMessageBox.information(self, "성공", "닉네임이 성공적으로 캡처되어 nickname.png로 저장되었습니다!")
        else:
            QMessageBox.warning(self, "실패", "화면 중앙 부근을 캡처하는 데 실패했습니다.")

    def toggle_bot(self):
        if self.bot_thread and self.bot_thread.isRunning():
            self.bot_thread.stop_bot()
            self.bot_thread.wait()
            self.toggle_btn.setText("▶ 사냥 시작")
            self.toggle_btn.setStyleSheet("background-color: #4CAF50; border: 3px solid #388E3C; border-radius: 15px;")
            return

        settings = {
            'map_file': self.map_combo.currentText(),
            'ai_file': self.ai_combo.currentText(),
            'key_attack': self.key_attack.text().lower(),
            'key_jump': self.key_jump.text().lower(),
            'key_teleport': self.key_teleport.text().lower(),
            'range_x': self.range_x.value(),
            'range_y': self.range_y.value(),
            'min_mobs': self.min_mobs.value(),
            'use_jump_atk': self.chk_jump_atk.isChecked(),
            'use_teleport': self.chk_teleport.isChecked()
        }
        
        self.bot_thread = BotThread(settings)
        self.bot_thread.start()
        
        self.toggle_btn.setText("⏹ 사냥 정지")
        self.toggle_btn.setStyleSheet("background-color: #f44336; border: 3px solid #c62828; border-radius: 15px;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainGUI()
    ex.show()
    sys.exit(app.exec())
