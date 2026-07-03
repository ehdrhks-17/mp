from PyQt6.QtCore import QThread, pyqtSignal
import time
from vision import VisionAgent
from controller import hold_key, let_go_key, tap_key

class BotThread(QThread):
    # UI로 상태나 경고 메시지를 보내기 위한 시그널
    log_signal = pyqtSignal(str)
    stop_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.vision = VisionAgent()
        
        # 순찰 방향 상태 ('left' or 'right')
        self.direction = 'right'
        self.patrol_start_time = time.time()
        # 한 방향으로 최대 이동할 시간 (예: 5초 후 방향 전환)
        self.patrol_duration = 5.0 

    def run(self):
        self.is_running = True
        self.log_signal.emit("사냥 봇 시작됨.")
        
        # 키 떼기 초기화
        let_go_key('left')
        let_go_key('right')

        while self.is_running:
            # 1. 최우선: 타 유저(빨간 점) 감지 (사용자 요청으로 제거됨)
            pass

            # 2. 몬스터(YOLO) 감지
            monsters = self.vision.find_monsters()
            
            if monsters:
                # 몬스터가 발견되면 이동을 멈추고 공격
                self.log_signal.emit(f"몬스터 발견! ({len(monsters)}마리)")
                let_go_key('left')
                let_go_key('right')
                
                # 목표 몬스터 결정 (화면 중앙에 가장 가까운 몬스터)
                screen_center_x = self.vision.monitor['width'] // 2
                target = min(monsters, key=lambda m: abs(m[0] - screen_center_x))
                mx, my, mw, mh, conf = target
                
                # 캐릭터가 화면 중앙에 있다고 가정
                # 몬스터가 중앙보다 왼쪽이면 왼쪽을 보고, 오른쪽이면 오른쪽을 봅니다.
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
                # 몬스터가 없으면 블라인드 순찰
                current_time = time.time()
                
                # 일정 시간 이동 후 방향 전환 (벽에 부딪히는 것을 방지하기 위함)
                if current_time - self.patrol_start_time > self.patrol_duration:
                    self.direction = 'left' if self.direction == 'right' else 'right'
                    self.patrol_start_time = current_time
                    self.log_signal.emit(f"방향 전환: {self.direction}")
                    
                    # 방향 전환 시 이전에 누르던 키 떼기
                    let_go_key('left')
                    let_go_key('right')
                
                # 현재 방향으로 이동
                hold_key(self.direction)
                
                # 이동하면서 가끔 점프 (장애물 회피용)
                if int(current_time) % 4 == 0:
                    tap_key('alt', 0.1)
                    
            time.sleep(0.1) # 루프 주기

        # 봇 종료 시 모든 키 뗌
        let_go_key('left')
        let_go_key('right')
        let_go_key('ctrl')
        self.log_signal.emit("사냥 봇 종료됨.")

    def stop_bot(self):
        self.is_running = False
