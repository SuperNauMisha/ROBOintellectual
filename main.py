import cv2
import cv2.aruco as aruco
import numpy as np
import paramiko
import time
import math

flag = True
debug = False
camera_control = False
video_stream = cv2.VideoCapture('http://root:admin@10.128.73.78/mjpg/video.mjpg')
cargos = np.array([[]])
target_ind = 0
hasTarget = False
target = [-1, -1]
targets = [9, 11, 5, 10]
print(cargos)
stop = True
def set_speed(lspeed, rspeed):
    if not debug:
        shell.send(f'{lspeed} {rspeed}\n')

if not debug:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.2.109', username='pi', password='rpi3')
    shell = ssh.invoke_shell()
    shell.send('cd workspace\n')
    shell.send('source venv/bin/activate\n')
    time.sleep(0.1)
    shell.send('python3 solution.py\n')
    set_speed(0, 0)
    time.sleep(1)

work_zone1 = (200, 0)
work_zone2 = (1050, 800)
keep_area = (96,  59, 255)
load_area = (105, 236, 226)
cargo = (103, 88, 233)
unload_area = (170, 130, 210)
dh, ds, dv = 26, 40, 19
dh2, ds2, dv2 = 19, 116, 45
dh3, ds3, dv3 = 9, 66, 68
dh4, ds4, dv4 = 9, 66, 68
x_click, y_click = -1, -1
arDict = {}
def calculate_center(corners):
    center_x = (corners[0][0][0] + corners[0][1][0] + corners[0][2][0] + corners[0][3][0]) // 4
    center_y = (corners[0][0][1] + corners[0][1][1] + corners[0][2][1] + corners[0][3][1]) // 4
    return [int(center_x), int(center_y)]

def vector_from_points(pos1, pos2):
    return np.array([int(pos2[0] - pos1[0]), int(pos2[1] - pos1[1])])


def vector_length(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

def replusion(pos):
    force = np.array([0.0, 0.0])
    for i in range(1, len(cargos)):
        R = vector_from_points(cargos[i], pos) / 400 # vector from charge to test charge
        if (not vector_length(R)): return [0, 0]
        force += (R / (vector_length(R) ** 2) * 4.5)  if vector_length(R) < 0.3 else 0
    x_f = int(force[0])
    y_f = int(force[1])
    force = np.array([x_f, y_f])
    return force

def attraction(pos, target):
    force = np.array([0.0, 0.0])
    if target[0] != -1:
        vect = vector_from_points(pos, target)
        force += vect / vector_length(vect) * 30
        x_f = int(force[0])
        y_f = int(force[1])
        force = np.array([x_f, y_f])
    return force

def calculate_angle(corners):
    # Углы маркера: [top-left, top-right, bottom-right, bottom-left]
    top_left = corners[0][0]
    top_right = corners[0][1]

    vector = top_right - top_left
    angle_rad = np.arctan2(vector[1], vector[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def findAruco(img, draw=True):
    global arDict, cargos
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)  # Используем новый метод
    arucoParam = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParam)  # Создаём детектор
    bbox, ids, _ = detector.detectMarkers(gray)  # Используем детектор
    if ids is not None and draw:
        cargos = np.array([[-1, -1]])
        for i, marker_id in enumerate(ids):
            color = (0, 255, 0)
            corners = bbox[i].astype(int)
            angle = calculate_angle(corners)
            center = calculate_center(corners)
            arDict[marker_id[0]] = [center, angle]
            if marker_id == 2 or marker_id == 3:
                color = (255, 0, 0)
                cv2.circle(img, center, 25, color, 3)
            if marker_id >= 4 and marker_id <= 11 and marker_id != targets[target_ind]:
                cent = np.array([center])
                cargos = np.concatenate((cargos, cent), axis=0)
            cv2.polylines(img, [corners], isClosed=True, color=color, thickness=2)
            cv2.putText(img, f"{marker_id[0]}", tuple(corners[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 0, 255), 2)
    return bbox, ids

def getXYaruco(id):
    global arDict
    try:
        return arDict[id][0]
    except KeyError:
        return None

def getANGaruco(id):
    global arDict
    try:
        return float(arDict[id][1])
    except KeyError:
        return (None, None)

def click(event, x, y, flags, param):
    global keep_area, load_area, unload_area, flag, x_click, y_click, stop
    if event == cv2.EVENT_RBUTTONDOWN:
    #     flag = not flag
        print(hsv[y][x])
    if event == cv2.EVENT_MBUTTONDOWN:
        load_area = hsv[y][x]
        print(load_area)
    if event == cv2.EVENT_LBUTTONDOWN:
        stop = not stop
        x_click = x
        y_click = y
        print("click", x_click, y_click)
        # unload_area = hsv[y][x]
        # print(unload_area)

def draw_contours(image, mask, num=2, color=(0, 255, 0), thickness=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # if num == 1:
    #     if len(contours) >= 2:
    #         biggest_contours = contours[:2]
    #         all_points = np.vstack(biggest_contours)
    #         x, y, w, h = cv2.boundingRect(all_points)
    #         cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    for i in range(min(num, len(contours))):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            # Рисуем прямоугольник на исходном изображении
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            #cv2.drawContours(image, [contour], -1, color, thickness)


def hue_dh_trackbar(val):
    global dh
    dh = val
def hue_ds_trackbar(val):
    global ds
    ds = val
def hue_dv_trackbar(val):
    global dv
    dv = val
def hue_dh2_trackbar(val):
    global dh2
    dh2 = val
def hue_ds2_trackbar(val):
    global ds2
    ds2 = val
def hue_dv2_trackbar(val):
    global dv2
    dv2 = val
def hue_dh3_trackbar(val):
    global dh3
    dh3 = val
def hue_ds3_trackbar(val):
    global ds3
    ds3 = val
def hue_dv3_trackbar(val):
    global dv3
    dv3 = val

def hue_dh4_trackbar(val):
    global dh4
    dh4 = val
def hue_ds4_trackbar(val):
    global ds4
    ds4 = val
def hue_dv4_trackbar(val):
    global dv4
    dv4 = val


if camera_control:
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("dh", "Trackbars", dh, 255, hue_dh_trackbar)
    cv2.createTrackbar("ds", "Trackbars", ds, 255, hue_ds_trackbar)
    cv2.createTrackbar("dv", "Trackbars", dv, 255, hue_dv_trackbar)
    cv2.createTrackbar("dh2", "Trackbars", dh2, 255, hue_dh2_trackbar)
    cv2.createTrackbar("ds2", "Trackbars", ds2, 255, hue_ds2_trackbar)
    cv2.createTrackbar("dv2", "Trackbars", dv2, 255, hue_dv2_trackbar)
    # cv2.createTrackbar("dh3", "Trackbars", dh3, 255, hue_dh3_trackbar)
    # cv2.createTrackbar("ds3", "Trackbars", ds3, 255, hue_ds3_trackbar)
    # cv2.createTrackbar("dv3", "Trackbars", dv3, 255, hue_dv3_trackbar)
    #
    # cv2.createTrackbar("dh4", "Trackbars", dh4, 255, hue_dh4_trackbar)
    # cv2.createTrackbar("ds4", "Trackbars", ds4, 255, hue_ds4_trackbar)
    # cv2.createTrackbar("dv4", "Trackbars", dv4, 255, hue_dv4_trackbar)



cv2.namedWindow("img")
cv2.setMouseCallback("img", click)


while True:
    img = video_stream.read()[1][work_zone1[1]:work_zone2[1], work_zone1[0]:work_zone2[0]]
    #img = cv2.imread("img.png")[work_zone1[1]:work_zone2[1], work_zone1[0]:work_zone2[0]]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bbox, ids = findAruco(img)
    lowKA = np.array([max(int(keep_area[0]) - dh, 0), max(int(keep_area[1]) - ds, 0), max(int(keep_area[2]) - dv, 0)])
    highKA = np.array([min(int(keep_area[0]) + dh, 255), min(int(keep_area[1]) + ds, 255), min(int(keep_area[2]) + dv, 255)])
    keep_area_mask = cv2.inRange(hsv, lowKA, highKA)
    lowLA = np.array([max(int(load_area[0]) - dh2, 0), max(int(load_area[1]) - ds2, 0), max(int(load_area[2]) - dv2, 0)])
    highLA = np.array([min(int(load_area[0]) + dh2, 255), min(int(load_area[1]) + ds2, 255), min(int(load_area[2]) + dv2, 255)])
    load_area_mask = cv2.inRange(hsv, lowLA, highLA)
    lowUA = np.array([max(int(unload_area[0]) - dh3, 0), max(int(unload_area[1]) - ds3, 0), max(int(unload_area[2]) - dv3, 0)])
    highUA = np.array(
        [min(int(unload_area[0]) + dh3, 255), min(int(unload_area[1]) + ds3, 255), min(int(unload_area[2]) + dv3, 255)])
    unload_area_mask = cv2.inRange(hsv, lowUA, highUA)

    lowCargo = np.array(
        [max(int(cargo[0]) - dh4, 0), max(int(cargo[1]) - ds4, 0), max(int(cargo[2]) - dv4, 0)])
    highCargo = np.array(
        [min(int(cargo[0]) + dh4, 255), min(int(cargo[1]) + ds4, 255), min(int(cargo[2]) + dv4, 255)])
    cargo_mask = cv2.inRange(hsv, lowCargo, highCargo)

    try:
        if getXYaruco(2)[0] != -1:
            xr, yr = getXYaruco(2)
            if hasTarget == False:
                target = getXYaruco(targets[target_ind])
                hasTarget = True


            if target[0] != -1 and not stop:
                dist_to_target = ((xr - target[0]) ** 2 + (yr - target[1]) ** 2) ** 0.5
                if dist_to_target < 40:
                    print("finish")
                    if len(targets) >= target_ind:
                        target_ind += 1
                    else:
                        stop = True

                    hasTarget = False
                for x in range(0, 40):
                    for y in range(0, 40):
                        r_pos = np.array([x * 25, y * 25])

                        vect_attr = attraction(r_pos, target)
                        vect_repl = replusion(r_pos)
                        sum_vect = vect_attr + vect_repl

                        cv2.arrowedLine(img, r_pos, (int(r_pos[0] + sum_vect[0]), int(r_pos[1] + sum_vect[1])), (0, 0, 255), 1)
                        # cv2.arrowedLine(img, r_pos, (int(r_pos[0] + vect_attr[0]), int(r_pos[1] + vect_attr[1])),
                        #                 (255, 0, 255), 1)


                        # cv2.arrowedLine(img, r_pos, (int(r_pos[0] + vect_move[0]), int(r_pos[1] + vect_move[1])), (255, 0, 0), 1)
                r_pos = np.array([xr, yr])
                vect_attr = attraction(r_pos, target)
                vect_repl = replusion(r_pos)
                sum_vect = vect_attr + vect_repl

                cv2.arrowedLine(img, r_pos, (int(r_pos[0] + sum_vect[0]), int(r_pos[1] + sum_vect[1])),
                                (0, 0, 255), 2)

                x_kat = int(sum_vect[0])
                y_kat = int(sum_vect[1])
                dist_err = (x_kat ** 2 + y_kat ** 2) ** 0.5
                ang_rad = np.arctan2(y_kat, x_kat)
                ang_deg = np.degrees(ang_rad)
                ang_r = getANGaruco(2)
                u = 0
                ang_err = 0
                if (ang_deg - ang_r + 90) % 360 != 0:
                    signs = [(ang_deg - ang_r + 90)/abs(ang_deg - ang_r + 90), (ang_deg - ang_r + 90 + 360)/abs(ang_deg - ang_r + 90 + 360), (ang_deg - ang_r + 90 - 360)/abs(ang_deg - ang_r + 90 - 360)]
                    errs = [abs(ang_deg - ang_r + 90), abs(ang_deg - ang_r + 90 + 360), abs(ang_deg - ang_r + 90 - 360)]
                    indx_err = errs.index(min(errs))
                    ang_err = min(errs) * signs[indx_err]
                if abs(ang_err) > 5 and dist_err > 100:
                    u = int(ang_err * 1)
                    set_speed(u, -u)
                else:
                    u = int(ang_err * 4)
                    u_dist = int(dist_err * 10)
                    set_speed(u_dist + u,u_dist - u)
            else:
                set_speed(0, 0)
    except Exception as err:
        print(err)

    if camera_control:
        cv2.imshow("keep_area_mask", keep_area_mask)
        cv2.imshow("load_area_mask", load_area_mask)
        cv2.imshow("unload_area_mask", unload_area_mask)
        cv2.imshow("cargo_mask", cargo_mask)

    draw_contours(img, keep_area_mask, num=7, color=(255, 255, 0), thickness=2)
    draw_contours(img, load_area_mask, num=1, color=(0, 255, 255), thickness=2)
    draw_contours(img, unload_area_mask, num=2, color=(255, 0, 255), thickness=2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) == 113:
        set_speed(0, 0)
        break

video_stream.release()
cv2.destroyAllWindows()