from PyQt6.QtCore import QThread, pyqtSignal
import time
import json
import os
from vision import VisionAgent
from controller import hold_key, let_go_key, tap_key

class BotThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.is_running = False
        self.settings = settings
        
        # 가중치 파일 로드
        ai_file = settings['ai_file']
        model_path = os.path.join(os.getcwd(), "mobs", ai_file) if ai_file != "기본 YOLO 모델 (yolov8n.pt)" else 'yolov8n.pt'
        self.vision = VisionAgent(yolo_model_path=model_path)
        
        # 맵 데이터 로드
        self.nodes = []
        map_file = settings['map_file']
        if map_file != "저장된 맵이 없습니다.":
            map_path = os.path.join("maps", map_file)
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    self.nodes = json.load(f)
                    
        self.box_size = 40

    def run(self):
        self.is_running = True
        self.log_signal.emit("사냥 봇 시작됨.")
        
        k_left = 'left'
        k_right = 'right'
        k_up = 'up'
        k_down = 'down'
        
        # 모든 이동키 떼기
        for k in [k_left, k_right, k_up, k_down]:
            let_go_key(k)

        while self.is_running:
            # 1. 내 캐릭터 정확한 위치 파악 (닉네임 기반)
            my_cx, my_cy = self.vision.find_nickname_pos()
            
            # 2. 몬스터(YOLO) 감지 및 사냥 조건 판별
            monsters = self.vision.find_monsters()
            monsters_in_range = []
            
            rx = self.settings['range_x']
            ry = self.settings['range_y']
            
            for m in monsters:
                mx, my, mw, mh, conf = m
                if abs(mx - my_cx) <= rx and abs(my - my_cy) <= ry:
                    monsters_in_range.append(m)
            
            if len(monsters_in_range) >= self.settings['min_mobs']:
                self.log_signal.emit(f"범위 내 목표물 발견! ({len(monsters_in_range)}마리)")
                let_go_key(k_left)
                let_go_key(k_right)
                
                target = min(monsters_in_range, key=lambda m: abs(m[0] - my_cx))
                mx, my, mw, mh, conf = target
                
                # 방향 전환
                if mx < my_cx:
                    tap_key(k_left, 0.1)
                else:
                    tap_key(k_right, 0.1)
                
                # 공격 루틴
                for _ in range(3):
                    if not self.is_running: break
                    if self.settings['use_jump_atk']:
                        tap_key(self.settings['key_jump'], 0.1)
                        time.sleep(0.05)
                    tap_key(self.settings['key_attack'], 0.1)
                    time.sleep(0.3)
                    
            else:
                # 3. 몬스터가 없거나 부족하면 미니맵 노드 기반 이동
                minimap_pos = self.vision.find_my_character()
                
                if minimap_pos:
                    mm_x, mm_y = minimap_pos
                    current_action = None
                    for node in self.nodes:
                        nx, ny = node['x'], node['y']
                        half = self.box_size // 2
                        if (nx - half) <= mm_x <= (nx + half) and (ny - half) <= mm_y <= (ny + half):
                            current_action = node['action']
                            break
                    
                    if current_action:
                        self.log_signal.emit(f"맵 노드 감지: {current_action}")
                        self.execute_node_action(current_action)
                else:
                    self.log_signal.emit("미니맵 캐릭터 잃어버림 (대기 중)")
                    
            time.sleep(0.1)

        for k in [k_left, k_right, k_up, k_down, self.settings['key_attack'], self.settings['key_jump'], self.settings['key_teleport']]:
            let_go_key(k)
        self.log_signal.emit("사냥 봇 종료됨.")

    def execute_node_action(self, action):
        let_go_key('left')
        let_go_key('right')
        let_go_key('up')
        let_go_key('down')
        
        k_jump = self.settings['key_jump']
        k_teleport = self.settings['key_teleport']
        use_tp = self.settings['use_teleport']
        
        if action == "walk_left":
            hold_key('left')
            if use_tp:
                tap_key(k_teleport, 0.1)
                time.sleep(0.3)
        elif action == "walk_right":
            hold_key('right')
            if use_tp:
                tap_key(k_teleport, 0.1)
                time.sleep(0.3)
        elif action == "jump_left":
            hold_key('left')
            tap_key(k_jump, 0.1)
        elif action == "jump_right":
            hold_key('right')
            tap_key(k_jump, 0.1)
        elif action == "rope_up":
            hold_key('up')
            tap_key(k_jump, 0.1)
            time.sleep(0.5)
        elif action == "jump_down":
            hold_key('down')
            tap_key(k_jump, 0.1)
            let_go_key('down')

    def stop_bot(self):
        self.is_running = False
