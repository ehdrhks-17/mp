import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from bot_logic import BotThread

class OverlayUI(QWidget):
    def __init__(self):
        super().__init__()
        self.bot_thread = None
        
        self.initUI()

    def initUI(self):
        # 윈도우 속성 설정: 항상 위, 프레임 없음, 포커스 받지 않음 (게임 창 유지)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        # 반투명 배경
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.8)
        
        # 레이아웃 및 스타일 설정
        layout = QVBoxLayout()
        
        self.log_label = QLabel("준비 완료", self)
        self.log_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 5px;")
        font = QFont("Malgun Gothic", 12)
        font.setBold(True)
        self.log_label.setFont(font)
        
        self.toggle_btn = QPushButton('▶ 사냥 시작', self)
        # 버튼을 안구 마우스로 클릭하기 쉽게 아주 크게 만듭니다.
        self.toggle_btn.setFixedSize(250, 150) 
        self.toggle_btn.setFont(QFont("Malgun Gothic", 24, QFont.Weight.Bold))
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                border: 3px solid #2E7D32;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_bot)

        layout.addWidget(self.log_label)
        layout.addWidget(self.toggle_btn)
        self.setLayout(layout)

        # 화면 우측 하단 쪽에 배치 (임시 좌표)
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('Maple YOLO Bot')

    def toggle_bot(self):
        if self.bot_thread is None or not self.bot_thread.is_running:
            # 시작
            self.bot_thread = BotThread()
            self.bot_thread.log_signal.connect(self.update_log)
            self.bot_thread.stop_signal.connect(self.on_bot_stopped)
            self.bot_thread.start()
            
            self.toggle_btn.setText("⏹ 사냥 정지")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border-radius: 20px;
                    border: 3px solid #c62828;
                }
            """)
        else:
            # 정지
            self.bot_thread.stop_bot()
            self.on_bot_stopped()

    def on_bot_stopped(self):
        self.toggle_btn.setText("▶ 사냥 시작")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                border: 3px solid #2E7D32;
            }
        """)

    def update_log(self, msg):
        self.log_label.setText(msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = OverlayUI()
    ex.show()
    sys.exit(app.exec())
