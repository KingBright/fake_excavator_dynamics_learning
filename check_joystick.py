import pygame
import time
import sys # To exit gracefully

# 初始化 Pygame 的所有模块（包括 joystick）
try:
    pygame.init()
    print("Pygame initialized successfully.")
except Exception as e:
    print(f"Error initializing Pygame: {e}")
    exit()

# 等待一小段时间，有时有助于设备检测
time.sleep(0.5)

# 再次初始化 joystick 模块（有时有帮助）
pygame.joystick.quit() # 先退出一次
try:
    pygame.joystick.init()
    print("Joystick module initialized.")
except Exception as e:
    print(f"Error initializing joystick module: {e}")
    pygame.quit()
    exit()

# 获取连接的手柄数量
joystick_count = pygame.joystick.get_count()
print(f"Number of joysticks detected: {joystick_count}")

if joystick_count == 0:
    print("No joysticks found. Please check connection, drivers, and permissions (especially macOS Input Monitoring).")
else:
    # 列出所有检测到的手柄
    for i in range(joystick_count):
        try:
            joystick = pygame.joystick.Joystick(i)
            joystick.init() # 初始化手柄以获取信息
            print(f"Joystick {i}:")
            print(f"  Name: {joystick.get_name()}")
            print(f"  ID: {joystick.get_id()}")
            print(f"  Instance ID: {joystick.get_instance_id()}")
            print(f"  Number of axes: {joystick.get_numaxes()}")
            print(f"  Number of buttons: {joystick.get_numbuttons()}")
            print(f"  Number of hats: {joystick.get_numhats()}")
            joystick.quit() # 退出初始化
        except Exception as e:
            print(f"Error getting info for joystick {i}: {e}")

# 退出 Pygame
pygame.quit()
print("Pygame quit.")
