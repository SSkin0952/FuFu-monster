# -*- coding: utf-8 -*-
from pylx16a.lx16a import *
from math import sin
import time

# 初始化串口（保持你原来的）
try:
    print("Initializing /dev/ttyUSB0...")
    LX16A.initialize("/dev/ttyUSB0", 0.1)
    print("/dev/ttyUSB0 initialized successfully")
except Exception as e:
    print(f"Failed to initialize /dev/ttyUSB0: {e}")
    quit()

servo_ids = [1,2,3,4,5,6,7,8]
servos = []

print("Testing servo connections...")

for sid in servo_ids:
    try:
        servo = LX16A(sid)
        angle = servo.get_physical_angle()
        print(f"Servo {sid} OK - angle {angle} deg")

        # 保留原始极限设置（pylx16a 内部会用它来校验 move）
        if sid == 7:
            servo.set_angle_limits(57, 109.2)
        elif sid == 8:
            servo.set_angle_limits(55, 106)
        else:
            servo.set_angle_limits(0, 240)

        servos.append(servo)
    except Exception as e:
        print(f"Servo {sid} error: {e}")
        servos.append(None)

# margin（度），用于生成“安全范围”以避免触及极限
MARGIN = 1.0

# 原始（library）极限（和上面设置一致）
HAIR_LIMITS = {
    7: (57.0, 109.2),
    8: (55.0, 106.0)
}

# 生成安全范围（将极限内收 MARGIN 度）
HAIR_SAFE_LIMITS = {}
for sid, (low, high) in HAIR_LIMITS.items():
    HAIR_SAFE_LIMITS[sid] = (low + MARGIN, high - MARGIN)

# O1 站立姿态：把 hair 的角度也内收到安全区间
O1_POSITION = {
    1: 117.36, 2: 118.56,
    3: 143.04, 4: 124.56,
    5: 130.32, 6: 138.24,
    # 用安全区间的中点或内收值，避免刚开始就在极限
    7: (HAIR_SAFE_LIMITS[7][0] + HAIR_SAFE_LIMITS[7][1]) / 2,   # mid of safe 7
    8: (HAIR_SAFE_LIMITS[8][0] + HAIR_SAFE_LIMITS[8][1]) / 2    # mid of safe 8
}

def clamp_angle_for_servo(sid, angle):
    """Clamp angle using known safe limits for hair, otherwise 0-240"""
    if sid in HAIR_SAFE_LIMITS:
        low, high = HAIR_SAFE_LIMITS[sid]
    else:
        low, high = (0.0, 240.0)
    # ensure float
    return max(min(float(angle), high), low)

def move_all(position_dict, duration=1.2):
    """平滑移动所有舵机，并在发送 move 前做角度夹紧（clamp）以避免超限错误"""
    steps = int(duration * 25)
    if steps < 1: steps = 1
    dt = duration / steps

    # 读取当前物理角度（fallback 到目标或 90）
    current = {}
    for i, s in enumerate(servos):
        sid = servo_ids[i]
        if s:
            try:
                current[sid] = s.get_physical_angle()
            except:
                current[sid] = position_dict.get(sid, 90.0)
        else:
            current[sid] = position_dict.get(sid, 90.0)

    # 插值并发送（每次发送前 clamp）
    for step in range(steps + 1):
        k = step / steps
        for i, s in enumerate(servos):
            sid = servo_ids[i]
            target = position_dict.get(sid, current[sid])
            start = current.get(sid, target)
            raw_angle = start + (target - start) * k
            angle = clamp_angle_for_servo(sid, raw_angle)
            if s:
                try:
                    s.move(angle)
                except Exception as e:
                    # 如果仍有异常，打印但继续尝试其他舵机
                    print(f"Warning: servo {sid} move error: {e}")
        time.sleep(dt)

# 先移动到 O1
print("\nMoving to O1 position (standing)...")
move_all(O1_POSITION, 1.5)
time.sleep(0.5)

# hair 来回摆动（在安全区间内）
print("\nStarting hair waving (safe limits + slower)...")
print("Press Ctrl+C to stop and return to O1")

try:
    speed = 0.15  # 更慢
    t = 0.0
    while True:
        t += speed
        for sid in [7, 8]:
            s = servos[sid - 1]
            if not s:
                continue
            low, high = HAIR_SAFE_LIMITS[sid]
            mid = (low + high) / 2.0
            amp = (high - low) / 2.0
            angle = mid + amp * sin(t)
            angle = clamp_angle_for_servo(sid, angle)
            try:
                s.move(angle)
            except Exception as e:
                print(f"Warning: servo {sid} move error during waving: {e}")
        time.sleep(0.06)

except KeyboardInterrupt:
    print("\nStopping... Returning to O1")
    move_all(O1_POSITION, 1.2)

print("Done.")