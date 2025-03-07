# Автоматизация разгрузочных процессов в складском помещении с помощью видеокамеры
## Основные алгоритмы:
- Поиск Аруко Маркера
- Высчитывание угла Аруко Маркера
- Отрисовка контура
- Пропорциональный регулятор 
- Построение маршрута по векторному полю
## Используемые библиотеки:
- OpenCV
- Numpy
- Paramiko
- Time
- Math


### Поиск аруко маркеров
Поиск осуществляется при помощи встроенного в  OpenCV модуля Aruko

```python
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
            if marker_id >= 4 and marker_id <= 11 and marker_id != target_id:  
                cent = np.array([center])  
                cargos = np.concatenate((cargos, cent), axis=0)  
            cv2.polylines(img, [corners], isClosed=True, color=color, thickness=2)  
            cv2.putText(img, f"{marker_id[0]}", tuple(corners[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 0, 255), 2)  
    return bbox, ids
```
### Высчитывание угла Аруко Марки
Высчитывание угла Аруко Метки осуществляется с помощью библиотеки Numpy  и таких функций как: arctan2(), degrees().
```python
def calculate_angle(corners):
    # Углы маркера: [top-left, top-right, bottom-right, bottom-left]
    top_left = corners[0][0]
    top_right = corners[0][1]

    vector = top_right - top_left
    angle_rad = np.arctan2(vector[1], vector[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg
```
### Отрисовка контуров
Для отрисовки контуров из кадра были выделины три основных цвета, на каждую из которых была наложена маскa и ориентируясь по границам  OpenCv отрисовывает границы. Также необходимо отметить, что, зачастую, из-за изменений света, маски приходилось перенастраивать, для удобства чего есть Trackbar на каждую из частей кадра.

- Отрисовка
```python
def draw_contours(image, mask, num=2, color=(0, 255, 0), thickness=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(min(num, len(contours))):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            # Рисуем прямоугольник на исходном изображении
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
....
....
....
    draw_contours(img, keep_area_mask, num=7, color=(255, 255, 0), thickness=2)
    draw_contours(img, load_area_mask, num=1, color=(0, 255, 255), thickness=2)
    draw_contours(img, unload_area_mask, num=2, color=(255, 0, 255), thickness=2)


```
- Trackbar
```python
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("dh", "Trackbars", dh, 255, hue_dh_trackbar)
    cv2.createTrackbar("ds", "Trackbars", ds, 255, hue_ds_trackbar)
    cv2.createTrackbar("dv", "Trackbars", dv, 255, hue_dv_trackbar)
    ......

```

### Пропорциональный регулятор
В целом принцип довольно прост. Работа регулятора разделяется на два случая. Если ошибка углов меньше 5, то в ход идет регулятор, предназначенный для поворота и движения робота, а если больше, то только для поворота.
```python 
            if abs(ang_err) > 5 and dist_err > 100:
                u = int(ang_err * 5)
                set_speed(u, -u)
            else:
                u = int(ang_err * 4)
                u_dist = int(dist_err * 10)
                set_speed(u_dist + u,u_dist - u)
                print("set_speed", u_dist + u,u_dist - u)

```

### Построение маршрута по векторному полю
На каждую точку карты можможно спроецировать вектор желаемого направления движения робота, складывающийся из двух векторов: одного, напрвленного в сторону желаемой точки, и других, направленных от препятствий к текущему положению. По этому вектору можно составить желаему траекторию движения с учетом препятствий.

```python
def replusion(pos):
    force = np.array([0.0, 0.0])
    for i in range(1, len(cargos)):
        R = vector_from_points(cargos[i], pos) / 400
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
....
....
....
        vect_attr = attraction(r_pos, target)
        vect_repl = replusion(r_pos)
        sum_vect = vect_attr + vect_repl


```
### Отправка скорости на моторы
Консоль ожидает получения скоростей моторов.

```python
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

....

def set_speed(lspeed, rspeed):
    if not debug:
        shell.send(f'{lspeed} {rspeed}\n')
```
###  Подключение из кода камеры по ssh к raspberry pi. 

```python
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.2.109', username='pi', password='rpi3')
    shell = ssh.invoke_shell()
    shell.send('cd workspace\n')
    shell.send('source venv/bin/activate\n')
    time.sleep(0.1)
    shell.send('python3 solution.py\n')
    set_speed(0, 0)
    time.sleep(0.5)
```
#
# [Видео нашего решения](https://drive.google.com/drive/folders/1--EMUVPgcaIZSfNFEPajci6DlPNGrIuc/ "Гугл Диск")
