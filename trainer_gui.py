import sys
import os
import subprocess
import threading
import time
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QLabel, QHBoxLayout, QLineEdit, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class TrainerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.collector_process = None
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        layout = QVBoxLayout()
        
        title = QLabel("🧠 인공지능 가중치 제작기")
        title.setFont(QFont("Malgun Gothic", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #E91E63;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 1단계
        step1 = QVBoxLayout()
        lbl1 = QLabel("[1단계] 몬스터 사진 수집")
        lbl1.setStyleSheet("color: #FFC107; font-size: 18px; font-weight: bold; margin-top: 10px;")
        step1.addWidget(lbl1)
        
        self.btn_collect = QPushButton("▶ 수집 시작 (2초마다 캡처)")
        self.btn_collect.setStyleSheet("background-color: #00BCD4; color: white; font-size: 18px; font-weight: bold; padding: 15px; border: 2px solid #0097A7;")
        self.btn_collect.clicked.connect(self.toggle_collect)
        step1.addWidget(self.btn_collect)
        layout.addLayout(step1)

        # 2단계
        step2 = QVBoxLayout()
        lbl2 = QLabel("[2단계] 몬스터 정답 찍기 (라벨링)")
        lbl2.setStyleSheet("color: #FFC107; font-size: 18px; font-weight: bold; margin-top: 10px;")
        step2.addWidget(lbl2)
        
        self.btn_label = QPushButton("🎯 드래그 라벨링 창 열기")
        self.btn_label.setStyleSheet("background-color: #8BC34A; color: white; font-size: 18px; font-weight: bold; padding: 15px; border: 2px solid #689F38;")
        self.btn_label.clicked.connect(self.run_labeler)
        step2.addWidget(self.btn_label)
        layout.addLayout(step2)

        # 3단계
        step3 = QVBoxLayout()
        lbl3 = QLabel("[3단계] 인공지능 학습 시작")
        lbl3.setStyleSheet("color: #FFC107; font-size: 18px; font-weight: bold; margin-top: 10px;")
        step3.addWidget(lbl3)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("파일 이름:"))
        self.name_input = QLineEdit("best")
        self.name_input.setStyleSheet("background: #444; padding: 5px; font-size: 16px;")
        name_layout.addWidget(self.name_input)
        name_layout.addWidget(QLabel(".pt"))
        step3.addLayout(name_layout)
        
        self.btn_train = QPushButton("🔥 인공지능(YOLO) 학습 시작")
        self.btn_train.setStyleSheet("background-color: #FF5722; color: white; font-size: 18px; font-weight: bold; padding: 15px; border: 2px solid #E64A19;")
        self.btn_train.clicked.connect(self.run_train)
        step3.addWidget(self.btn_train)
        layout.addLayout(step3)

        self.setLayout(layout)
        self.setGeometry(150, 150, 450, 500)
        self.setWindowTitle('Trainer GUI')

    def toggle_collect(self):
        if self.collector_process is None:
            # 수집 시작
            self.collector_process = subprocess.Popen([sys.executable, "data_collector.py"])
            self.btn_collect.setText("⏹ 수집 정지")
            self.btn_collect.setStyleSheet("background-color: #f44336; color: white; font-size: 18px; font-weight: bold; padding: 15px; border: 2px solid #c62828;")
        else:
            # 수집 종료
            self.collector_process.terminate()
            self.collector_process = None
            self.btn_collect.setText("▶ 수집 시작 (2초마다 캡처)")
            self.btn_collect.setStyleSheet("background-color: #00BCD4; color: white; font-size: 18px; font-weight: bold; padding: 15px; border: 2px solid #0097A7;")

    def run_labeler(self):
        subprocess.Popen([sys.executable, "labeler.py"])

    def run_train(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "경고", "파일 이름을 입력하세요!")
            return
            
        QMessageBox.information(self, "알림", "학습을 시작합니다. 프롬프트(CMD) 창에서 진행 상황을 확인하세요.\n(사양에 따라 몇 분 정도 소요됩니다.)")
        # 학습 시작 (CMD 창 띄움)
        os.system(f'start cmd /k "{sys.executable} train.py"')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TrainerGUI()
    ex.show()
    sys.exit(app.exec())
