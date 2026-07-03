from PyQt6.QtCore import QThread, pyqtSignal
import time
import json
import os
from vision import VisionAgent
from controller import hold_key, let_go_key, tap_key

class BotThread(QThread):
    log_signal = pyqtSignal(str)
    stop_signal = pyqtSignal()

    def __init__(self, map_file, ai_file):
        super().__init__()
        self.is_running = False
        self.map_file = map_file
        self.ai_file = ai_file
        
        # 가중치 파일 로드
        model_path = os.path.join(os.getcwd(), ai_file) if ai_file != "기본 YOLO 모델 (yolov8n.pt)" else 'yolov8n.pt'
        self.vision = VisionAgent(yolo_model_path=model_path)
        
        # 맵 데이터(노드) 로드
        self.nodes = []
        if map_file != "저장된 맵이 없습니다.":
            map_path = os.path.join("maps", map_file)
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    self.nodes = json.load(f)
                    
        self.box_size = 40 # map_editor.py와 동일한 크기

    def run(self):
        self.is_running = True
        self.log_signal.emit("사냥 봇 시작됨.")
        
        if not self.nodes:
            self.log_signal.emit("경고: 불러온 맵 노드가 없습니다! 제자리에서 공격만 합니다.")

        # 모든 키 떼기 초기화
        let_go_key('left')
        let_go_key('right')
        let_go_key('up')
        let_go_key('down')

        while self.is_running:
            # 1. 몬스터(YOLO) 감지 우선순위
            monsters = self.vision.find_monsters()
            
            if monsters:
                self.log_signal.emit(f"몬스터 발견! ({len(monsters)}마리)")
                let_go_key('left')
                let_go_key('right')
                
                screen_center_x = self.vision.monitor['width'] // 2
                target = min(monsters, key=lambda m: abs(m[0] - screen_center_x))
                mx, my, mw, mh, conf = target
                
                if mx < screen_center_x:
                    tap_key('left', 0.1)
                else:
                    tap_key('right', 0.1)
                
                # 공격 연사
                for _ in range(3):
                    if not self.is_running: break
                    tap_key('ctrl', 0.1)
                    time.sleep(0.3)
                    
            else:
                # 2. 몬스터가 없으면 미니맵 노드 기반 이동
                # 내 캐릭터(노란 점)의 미니맵 좌표 찾기
                my_pos = self.vision.find_my_character()
                
                if my_pos:
                    my_x, my_y = my_pos
                    # 현재 내가 어느 노드(박스) 안에 있는지 확인
                    current_action = None
                    for node in self.nodes:
                        nx, ny = node['x'], node['y']
                        half = self.box_size // 2
                        # 내 좌표가 노드 박스 안에 있다면
                        if (nx - half) <= my_x <= (nx + half) and (ny - half) <= my_y <= (ny + half):
                            current_action = node['action']
                            break
                    
                    if current_action:
                        self.log_signal.emit(f"맵 노드 감지: {current_action}")
                        self.execute_node_action(current_action)
                    else:
                        # 박스 밖에 있으면 이전에 누르던 키를 계속 유지 (관성)
                        pass
                else:
                    self.log_signal.emit("미니맵에서 캐릭터를 찾을 수 없습니다.")
                    
            time.sleep(0.1)

        # 봇 종료 시 모든 키 뗌
        let_go_key('left')
        let_go_key('right')
        let_go_key('up')
        let_go_key('down')
        let_go_key('ctrl')
        self.log_signal.emit("사냥 봇 종료됨.")

    def execute_node_action(self, action):
        """노드에 정의된 행동을 수행합니다."""
        let_go_key('left')
        let_go_key('right')
        let_go_key('up')
        let_go_key('down')
        
        if action == "walk_left":
            hold_key('left')
        elif action == "walk_right":
            hold_key('right')
        elif action == "jump_left":
            hold_key('left')
            tap_key('alt', 0.1)
        elif action == "jump_right":
            hold_key('right')
            tap_key('alt', 0.1)
        elif action == "rope_up":
            hold_key('up')
            tap_key('alt', 0.1) # 줄에 매달리기 위해 점프 동반
            time.sleep(0.5) # 줄을 탈 시간 부여
        elif action == "jump_down":
            hold_key('down')
            tap_key('alt', 0.1)
            let_go_key('down')

    def stop_bot(self):
        self.is_running = False
