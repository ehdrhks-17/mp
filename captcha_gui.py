import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import os
import threading
import time

from captcha_simulator import CaptchaEngine

class CaptchaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Captcha Simulator & Dataset Generator")
        self.root.geometry("1100x650")
        
        self.engine = CaptchaEngine(width=800, height=600)
        self.is_playing = False
        
        self.setup_ui()
        self.reset_simulation()
        
    def setup_ui(self):
        # Left Panel (Controls)
        control_frame = ttk.Frame(self.root, padding=10, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(control_frame, text="Shape:").pack(anchor=tk.W, pady=(0, 5))
        self.shape_var = tk.StringVar(value="star")
        shape_combo = ttk.Combobox(control_frame, textvariable=self.shape_var, values=["star", "circle", "triangle", "square"], state="readonly")
        shape_combo.pack(fill=tk.X, pady=(0, 15))
        shape_combo.bind("<<ComboboxSelected>>", lambda e: self.reset_simulation())
        
        ttk.Label(control_frame, text="Fake Count:").pack(anchor=tk.W)
        self.fake_var = tk.IntVar(value=10)
        fake_slider = ttk.Scale(control_frame, from_=0, to=30, orient=tk.HORIZONTAL, variable=self.fake_var, command=lambda v: self.reset_simulation())
        fake_slider.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(control_frame, text="Target Transparency (0=White, 1=Invisible):").pack(anchor=tk.W)
        self.phase_var = tk.DoubleVar(value=0.0)
        phase_slider = ttk.Scale(control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.phase_var)
        phase_slider.pack(fill=tk.X, pady=(0, 20))
        
        # Play/Pause
        self.play_btn = ttk.Button(control_frame, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(fill=tk.X, pady=5)
        
        # Reset
        ttk.Button(control_frame, text="🔄 Reset", command=self.reset_simulation).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # Dataset Generation
        ttk.Label(control_frame, text="Dataset Generation").pack(anchor=tk.W, pady=5)
        self.gen_btn = ttk.Button(control_frame, text="💾 Generate 1000 Frames", command=self.generate_dataset)
        self.gen_btn.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue").pack(anchor=tk.W, pady=10)
        
        # Right Panel (Preview)
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="black")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Animation loop
        self.root.after(30, self.update_frame)
        
    def reset_simulation(self, *args):
        self.engine.initialize(shape_type=self.shape_var.get(), fake_count=self.fake_var.get())
        if not self.is_playing:
            self.draw_current_frame()
            
    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="⏸ Pause" if self.is_playing else "▶ Play")
            
    def draw_current_frame(self):
        frame, bbox = self.engine.render(transition_phase=self.phase_var.get())
        
        # Draw bounding box for visualization
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
    def update_frame(self):
        if self.is_playing:
            self.engine.update()
            self.draw_current_frame()
        self.root.after(30, self.update_frame)
        
    def generate_dataset(self):
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        self.gen_btn.config(state=tk.DISABLED)
        self.status_var.set("Generating...")
        
        # Run in thread to not freeze UI
        threading.Thread(target=self._generation_task, daemon=True).start()
        
    def _generation_task(self):
        output_dir = "dataset"
        img_dir = os.path.join(output_dir, "images", "train")
        lbl_dir = os.path.join(output_dir, "labels", "train")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        num_frames = 1000
        shape_type = self.shape_var.get()
        
        # Reset and run background simulation
        self.engine.initialize(shape_type=shape_type, fake_count=self.fake_var.get())
        
        for i in range(num_frames):
            self.engine.update()
            
            # Vary transparency randomly during dataset generation
            phase = i / num_frames if i < num_frames // 2 else 1.0 # First half fades, second half fully transparent
            frame, bbox = self.engine.render(transition_phase=phase)
            
            if bbox:
                bx, by, bw, bh = bbox
                # YOLO format
                cx = (bx + bw / 2) / self.engine.width
                cy = (by + bh / 2) / self.engine.height
                nw = bw / self.engine.width
                nh = bh / self.engine.height
                
                img_name = f"sim_{shape_type}_{i:05d}.jpg"
                lbl_name = f"sim_{shape_type}_{i:05d}.txt"
                
                cv2.imwrite(os.path.join(img_dir, img_name), frame) # Save clean frame (no green box)
                with open(os.path.join(lbl_dir, lbl_name), "w") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    
            if i % 50 == 0:
                self.root.after(0, lambda curr=i: self.status_var.set(f"Generating: {curr}/{num_frames}"))
                
        self.root.after(0, self._generation_done)
        
    def _generation_done(self):
        self.status_var.set("Dataset Generation Complete!")
        self.gen_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = CaptchaGUI(root)
    root.mainloop()
