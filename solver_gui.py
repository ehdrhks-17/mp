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
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top panel for controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Status: STOPPED (Press F8 to Toggle)")
        ttk.Label(top_frame, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
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

    def bot_loop(self):
        with mss.mss() as sct:
            # 기본적으로 전체 화면 캡처. 원한다면 bbox 지정 가능
            monitor = sct.monitors[1] 
            
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
                    # 화면 캡처
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[:, :, :3] # BGRA to BGR
                    
                    # 봇 프로세스 실행
                    self.solver.process_frame(frame, monitor["left"], monitor["top"])
                    
                time.sleep(0.01)

if __name__ == "__main__":
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()
