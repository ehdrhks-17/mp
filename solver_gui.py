import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import keyboard
import time
import mss
import numpy as np

from captcha_solver import CaptchaSolver

class SolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Occlusion-Shield Bot (CSRT + Kalman)")
        self.root.geometry("1000x500")
        
        self.solver = CaptchaSolver(debug_callback=self.on_frame_ready)
        self.is_running = False
        self.roi_dict = None  # None이면 전체 화면
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top panel for controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Status: STOPPED (Press F8 to Toggle)")
        ttk.Label(top_frame, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        self.roi_btn = ttk.Button(top_frame, text="🔍 캡처 영역 지정하기 (크롭)", command=self.select_roi)
        self.roi_btn.pack(side=tk.RIGHT, padx=10)
        
        # Bottom panel for video feeds
        video_frame = ttk.Frame(self.root)
        video_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Left feed: Main tracking view
        left_frame = ttk.Frame(video_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(left_frame, text="Main Tracking View (Red: Predict, Green: Measure)").pack()
        self.canvas_main = tk.Canvas(left_frame, bg="black", width=400, height=300)
        self.canvas_main.pack(fill=tk.BOTH, expand=True)
        
        # Right feed: Preprocessed view
        right_frame = ttk.Frame(video_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(right_frame, text="CLAHE Preprocessed View (Fed to Tracker)").pack()
        self.canvas_prep = tk.Canvas(right_frame, bg="black", width=400, height=300)
        self.canvas_prep.pack(fill=tk.BOTH, expand=True)
        
        # Start background hook
        threading.Thread(target=self.bot_loop, daemon=True).start()

    def on_frame_ready(self, debug_img, prep_img):
        # Resize for display
        debug_img = cv2.resize(debug_img, (480, 360))
        prep_img = cv2.resize(prep_img, (480, 360))
        
        debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        prep_rgb = cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB)
        
        img_main = ImageTk.PhotoImage(image=Image.fromarray(debug_rgb))
        img_prep = ImageTk.PhotoImage(image=Image.fromarray(prep_rgb))
        
        # Thread-safe UI update
        def update_canvas():
            self.canvas_main.create_image(0, 0, image=img_main, anchor=tk.NW)
            self.canvas_main.image = img_main
            
            self.canvas_prep.create_image(0, 0, image=img_prep, anchor=tk.NW)
            self.canvas_prep.image = img_prep
            
        self.root.after(0, update_canvas)

    def select_roi(self):
        # 봇이 돌고 있다면 잠시 정지
        was_running = self.is_running
        self.is_running = False
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)[:, :, :3]
            
            # 해상도가 너무 높으면 화면에 안 들어올 수 있으므로 cv2.selectROI 사용 전 확인
            # OpenCV 창에서 마우스 드래그로 영역 선택 (Enter나 Space를 누르면 완료)
            r = cv2.selectROI("Select Captcha Region (Press Enter to confirm)", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select Captcha Region (Press Enter to confirm)")
            
            x, y, w, h = r
            if w > 0 and h > 0:
                self.roi_dict = {"top": monitor["top"] + y, "left": monitor["left"] + x, "width": w, "height": h}
                print(f"ROI selected: {self.roi_dict}")
            else:
                print("ROI selection cancelled. Using full screen.")
                self.roi_dict = None
                
        self.is_running = was_running

    def bot_loop(self):
        with mss.mss() as sct:
            while True:
                if keyboard.is_pressed('f8'):
                    self.is_running = not self.is_running
                    status = "RUNNING" if self.is_running else "STOPPED"
                    color = "green" if self.is_running else "red"
                    self.root.after(0, lambda s=status: self.status_var.set(f"Status: {s} (Press F8 to Toggle)"))
                    
                    if not self.is_running:
                        self.solver.state = "SEARCHING"
                    time.sleep(0.3) # Debounce
                    
                if self.is_running:
                    # 화면 캡처 (선택된 ROI가 있으면 그것만, 없으면 모니터 전체)
                    monitor = self.roi_dict if self.roi_dict else sct.monitors[1]
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[:, :, :3] # BGRA to BGR
                    
                    # 봇 프로세스 실행
                    self.solver.process_frame(frame, monitor["left"], monitor["top"])
                    
                time.sleep(0.01)

if __name__ == "__main__":
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()
