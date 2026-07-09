import ctypes
import time

# DirectInput SendInput C Structs
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I)
    ]

# Map standard key names to DirectInput scan codes
# Common keys for MapleStory
KEYS = {
    'ctrl': 0x1D,  # Attack
    'alt': 0x38,   # Jump
    'left': 0xCB,  # Arrow Left
    'right': 0xCD, # Arrow Right
    'up': 0xC8,    # Arrow Up
    'down': 0xD0,  # Arrow Down
}

def press_key(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def tap_key(key_name, duration=0.05):
    """지정된 키를 짧게 누릅니다."""
    if key_name not in KEYS:
        print(f"Unknown key: {key_name}")
        return
    
    code = KEYS[key_name]
    press_key(code)
    time.sleep(duration)
    release_key(code)

def hold_key(key_name):
    """지정된 키를 누르고 유지합니다."""
    if key_name in KEYS:
        press_key(KEYS[key_name])

def let_go_key(key_name):
    """유지 중인 키를 뗍니다."""
    if key_name in KEYS:
        release_key(KEYS[key_name])

def move_mouse(x, y):
    """마우스를 절대 좌표 (x, y)로 이동합니다."""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    # 0x8000 = MOUSEEVENTF_ABSOLUTE, 0x0001 = MOUSEEVENTF_MOVE
    # 좌표 변환: 0~65535로 스케일링 필요
    # 화면 해상도를 1920x1080으로 가정하거나, win32api에서 받아올 수 있음.
    # 단순화를 위해 SetCursorPos 사용 후 SendInput 빈 이벤트로 후킹 통과 유도.
    ctypes.windll.user32.SetCursorPos(x, y)
    
def click_mouse():
    """마우스 왼쪽 버튼 클릭."""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    # 0x0002 = LEFTDOWN, 0x0004 = LEFTUP
    ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    time.sleep(0.05)
    
    ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

if __name__ == "__main__":
    # Test block
    print("Testing Ctrl tap in 3 seconds...")
    time.sleep(3)
    tap_key('ctrl')
    print("Test complete.")
