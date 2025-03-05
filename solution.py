from ctypes.wintypes import tagSIZE

import cv2
import threading
from dotenv import load_dotenv
from nto.final import Task
from my_teleop import MyTeleop


load_dotenv('.env')

lspeed = 0
rspeed = 0
def change_speed():
    global lspeed, rspeed, edited
    teleop = MyTeleop()
    while True:
        try:
            l, r = input().split()
            lspeed = int(l)
            rspeed = int(r)
            teleop.send_velocity(lspeed, rspeed)
        except KeyboardInterrupt:
            break
    teleop.send_velocity(0, 0)

            

def solve():
    # global edited
    # thread = threading.Thread(target=change_speed)
    # thread.daemon = True
    # thread.start()
    change_speed()
    # # Запускаем цикл обновления изображения с камеры
    # scene = [None]
    # while True:
    #     try:
    #         if edited:
    #             #teleop.send_velocity(lspeed, rspeed)
    #             edited = False
    #         # scene = task.getTaskScene()
    #     except KeyboardInterrupt:
    #         teleop.send_velocity(0, 0)
    #         break

    # # Завершаем работу: освобождаем ресурсы камеры и закрываем окна
    # task.stop()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    solve()
