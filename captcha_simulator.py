import cv2
import numpy as np
import random
import math

class ShapeObject:
    def __init__(self, x, y, size, shape_type="star", is_target=False):
        self.x = x
        self.y = y
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        
        # Ensure minimum speed
        if abs(self.vx) < 1.0: self.vx = 2.0 if self.vx > 0 else -2.0
        if abs(self.vy) < 1.0: self.vy = 2.0 if self.vy > 0 else -2.0
        
        self.size = size
        self.shape_type = shape_type
        self.is_target = is_target
        
        self.angle = random.uniform(0, 360)
        self.angular_velocity = random.uniform(-3, 3)

    def update(self, width, height):
        self.x += self.vx
        self.y += self.vy
        self.angle = (self.angle + self.angular_velocity) % 360
        
        # Bounce off edges
        if self.x < 0:
            self.x = 0
            self.vx *= -1
        elif self.x + self.size > width:
            self.x = width - self.size
            self.vx *= -1
            
        if self.y < 0:
            self.y = 0
            self.vy *= -1
        elif self.y + self.size > height:
            self.y = height - self.size
            self.vy *= -1

class CaptchaEngine:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.objects = []
        self.target = None
        
        # 원본 타일 배경 로드
        bg_img = cv2.imread('bg.png')
        if bg_img is not None:
            self.bg = cv2.resize(bg_img, (width, height))
        else:
            self.bg = self._generate_bg()
            
    def _generate_bg(self):
        bg = np.random.randint(50, 150, (self.height, self.width, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (21, 21), 0)
        noise = np.random.randint(-20, 20, (self.height, self.width, 3), dtype=np.int8)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return bg

    def _get_star_points(self, center_x, center_y, outer_radius, inner_radius, num_points=5, angle_offset=0):
        points = []
        angle_step = math.pi / num_points
        for i in range(num_points * 2):
            radius = outer_radius if i % 2 == 0 else inner_radius
            angle = i * angle_step + angle_offset - math.pi/2
            x = center_x + int(math.cos(angle) * radius)
            y = center_y + int(math.sin(angle) * radius)
            points.append([x, y])
        return np.array([points], dtype=np.int32)

    def _get_shape_mask(self, obj):
        size = obj.size
        mask = np.zeros((size, size), dtype=np.uint8)
        cx, cy = size//2, size//2
        angle_rad = math.radians(obj.angle)
        
        if obj.shape_type == "star":
            pts = self._get_star_points(cx, cy, size//2 - 2, size//4, 5, angle_rad)
            cv2.fillPoly(mask, pts, 255)
        elif obj.shape_type == "circle":
            cv2.circle(mask, (cx, cy), size//2 - 2, 255, -1)
        elif obj.shape_type == "triangle":
            pts = self._get_star_points(cx, cy, size//2 - 2, size//4, 3, angle_rad)
            cv2.fillPoly(mask, pts, 255)
        else: # square
            rect_size = int((size//2) * 0.8)
            cv2.rectangle(mask, (cx-rect_size, cy-rect_size), (cx+rect_size, cy+rect_size), 255, -1)
            
        if obj.shape_type in ["square", "circle"]:
            M = cv2.getRotationMatrix2D((cx, cy), obj.angle, 1.0)
            mask = cv2.warpAffine(mask, M, (size, size))
            
        return mask

    def initialize(self, shape_type="star", fake_count=10):
        self.objects = []
        # Create fakes
        for _ in range(fake_count):
            size = random.randint(50, 90)
            x = random.randint(0, self.width - size)
            y = random.randint(0, self.height - size)
            self.objects.append(ShapeObject(x, y, size, shape_type, is_target=False))
            
        # Create target
        size = random.randint(60, 80)
        x = random.randint(0, self.width - size)
        y = random.randint(0, self.height - size)
        self.target = ShapeObject(x, y, size, shape_type, is_target=True)
        self.objects.append(self.target)

    def update(self):
        for obj in self.objects:
            obj.update(self.width, self.height)

    def render(self, transition_phase=0.0):
        frame = self.bg.copy()
        target_bbox = None
        
        # Render fakes
        for obj in self.objects:
            if obj.is_target: continue
            mask = self._get_shape_mask(obj)
            x, y = int(obj.x), int(obj.y)
            self._apply_transparent_effect(frame, mask, x, y)
            
        # Render target
        if self.target:
            mask = self._get_shape_mask(self.target)
            x, y = int(self.target.x), int(self.target.y)
            success, bbox = self._apply_real_target_effect(frame, mask, x, y, transition_phase)
            if success:
                target_bbox = bbox
                
        return frame, target_bbox

    def _apply_transparent_effect(self, bg, mask, x, y):
        h, w = mask.shape
        bg_roi = bg[y:y+h, x:x+w]
        if bg_roi.shape[:2] != mask.shape: return
            
        result_roi = bg_roi.copy().astype(np.float32)
        
        # 엠보싱(Emboss) 및 굴절 효과 (실제 원본 영상과 똑같은 테두리 음영)
        # 1픽셀씩 이동시켜서 하이라이트(밝은 테두리)와 섀도우(어두운 테두리) 생성
        M_shift_up_left = np.float32([[1, 0, -2], [0, 1, -2]])
        M_shift_down_right = np.float32([[1, 0, 2], [0, 1, 2]])
        
        mask_up_left = cv2.warpAffine(mask, M_shift_up_left, (w, h))
        mask_down_right = cv2.warpAffine(mask, M_shift_down_right, (w, h))
        
        highlight = cv2.subtract(mask, mask_up_left) > 0
        shadow = cv2.subtract(mask, mask_down_right) > 0
        inner = (mask > 0) & (~highlight) & (~shadow)
        
        # 외곽선 하이라이트(빛 반사) 및 그림자
        result_roi[highlight] = result_roi[highlight] * 1.5 + 40
        result_roi[shadow] = result_roi[shadow] * 0.5
        
        # 내부는 미세하게 어둡고 굴절된 느낌
        result_roi[inner] = result_roi[inner] * 0.95
        
        result_roi = np.clip(result_roi, 0, 255).astype(np.uint8)
        bg[y:y+h, x:x+w] = result_roi

    def _apply_real_target_effect(self, bg, mask, x, y, transition_phase):
        h, w = mask.shape
        bg_roi = bg[y:y+h, x:x+w]
        if bg_roi.shape[:2] != mask.shape: return False, None
            
        result_roi = bg_roi.copy().astype(np.float32)
        
        M_shift_up_left = np.float32([[1, 0, -2], [0, 1, -2]])
        M_shift_down_right = np.float32([[1, 0, 2], [0, 1, 2]])
        
        mask_up_left = cv2.warpAffine(mask, M_shift_up_left, (w, h))
        mask_down_right = cv2.warpAffine(mask, M_shift_down_right, (w, h))
        
        highlight = cv2.subtract(mask, mask_up_left) > 0
        shadow = cv2.subtract(mask, mask_down_right) > 0
        inner = (mask > 0) & (~highlight) & (~shadow)
        
        white_blend = 1.0 - transition_phase
        
        base_highlight = result_roi[highlight] * 1.5 + 40
        result_roi[highlight] = base_highlight * transition_phase + 255 * white_blend
        
        base_shadow = result_roi[shadow] * 0.5
        result_roi[shadow] = base_shadow * transition_phase + 255 * white_blend
        
        base_inner = result_roi[inner] * 0.95
        result_roi[inner] = base_inner * transition_phase + 255 * white_blend
        
        result_roi = np.clip(result_roi, 0, 255).astype(np.uint8)
        bg[y:y+h, x:x+w] = result_roi
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            rx, ry, rw, rh = cv2.boundingRect(contours[0])
            return True, (x + rx, y + ry, rw, rh)
        return False, None
