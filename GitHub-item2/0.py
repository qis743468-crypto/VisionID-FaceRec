import sys
import os
import time
import json
import cv2 as cv
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QMessageBox, 
                             QDialog, QLineEdit, QFormLayout, QFrame)
from PySide6.QtGui import QImage, QPixmap, QColor, QFont
from PySide6.QtCore import Qt, QTimer

# ==================== 全局变量与分类器 ====================
cap = None
timer = None
live_timer = None
scan_line_y = 0  # 扫描线位置

USER_DATA_FILE = "./user_data.json"
EYE_CASCADE_PATH = r"D:\kdlstudy\Python\haar\haarcascade_eye_tree_eyeglasses.xml"
FACE_CASCADE_PATH = r"D:\kdlstudy\Python\haar\haarcascade_frontalface_alt2.xml"

# 分类器载入
facecascade = cv.CascadeClassifier(FACE_CASCADE_PATH if os.path.exists(FACE_CASCADE_PATH) else "")
eye_cascade = cv.CascadeClassifier(EYE_CASCADE_PATH if os.path.exists(EYE_CASCADE_PATH) else "")
recognizer = cv.face.LBPHFaceRecognizer.create()

# 活体检测参数
is_live_detect = False
blink_count = 0
eye_closed_flag = False
blink_need = 2
live_start_time = 0
live_timeout = 5.0


# ==================== JSON 用户信息管理 ====================
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_info(user_id, name, age, unit):
    data = load_user_data()
    data[str(user_id)] = {
        "name": name,
        "age": age,
        "unit": unit
    }
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ==================== 录入信息多字段弹窗 ====================
class UserInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("录入身份档案")
        self.setFixedSize(320, 240)
        self.setStyleSheet("""
            QDialog { background-color: #111827; }
            QLabel { color: #00F0FF; font-weight: bold; }
            QLineEdit { 
                background-color: #1F2937; color: #FFFFFF; 
                border: 1px solid #374151; border-radius: 5px; padding: 6px; 
            }
            QPushButton { 
                background-color: #10B981; color: white; border-radius: 5px; 
                padding: 8px; font-weight: bold; 
            }
            QPushButton:hover { background-color: #059669; }
        """)

        layout = QFormLayout(self)
        self.id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.age_input = QLineEdit()
        self.unit_input = QLineEdit()

        layout.addRow("数字编号(ID):", self.id_input)
        layout.addRow("姓名:", self.name_input)
        layout.addRow("年龄:", self.age_input)
        layout.addRow("工作单位:", self.unit_input)

        self.submit_btn = QPushButton("保存信息")
        self.submit_btn.clicked.connect(self.accept)
        layout.addRow(self.submit_btn)

    def get_data(self):
        return (
            self.id_input.text().strip(),
            self.name_input.text().strip(),
            self.age_input.text().strip(),
            self.unit_input.text().strip()
        )


# ==================== 现代科技风 QSS 样式表 ====================
DARK_STYLE = """
QMainWindow { background-color: #0B0F19; }

QFrame#cardFrame {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
}

QLabel#titleLabel {
    color: #00F0FF; font-size: 18px; font-weight: bold;
    font-family: "Microsoft YaHei", sans-serif;
}

QLabel#statusLabel {
    color: #9CA3AF; font-size: 13px; background-color: #1F2937;
    border-radius: 6px; padding: 5px 10px;
}

QLabel#camView {
    background-color: #030712; border: 2px solid #00F0FF; border-radius: 10px;
}

/* 右侧身份信息卡片 */
QFrame#profileCard {
    background: #162032;
    border: 1px solid #00F0FF;
    border-radius: 10px;
}

QLabel.infoTitle { color: #9CA3AF; font-size: 12px; }
QLabel.infoValue { color: #FFFFFF; font-size: 15px; font-weight: bold; }
QLabel#infoName { color: #00F0FF; font-size: 20px; font-weight: bold; }

QPushButton {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    color: #E2E8F0; border: 1px solid #334155; border-radius: 6px;
    padding: 10px; font-size: 13px; font-weight: bold;
}
QPushButton:hover { background: #334155; border-color: #00F0FF; color: #00F0FF; }
QPushButton#btnLive { border-color: #F59E0B; color: #FBBF24; }
QPushButton#btnRecog { border-color: #10B981; color: #34D399; }
"""


# ==================== 主界面逻辑 ====================
class FaceRecognitionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高阶AI人脸识别与身份比对系统")
        self.resize(1100, 680)
        self.setStyleSheet(DARK_STYLE)

        # 新增状态标志：标记当前是否已成功识别并锁定结果
        self.is_recognized = False

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)

        # --- 左侧：视频显示区域 ---
        cam_card = QFrame()
        cam_card.setObjectName("cardFrame")
        cam_layout = QVBoxLayout(cam_card)

        top_bar = QHBoxLayout()
        self.title_label = QLabel("实时视频监控 & 动态识别分析区")
        self.title_label.setObjectName("titleLabel")
        self.status_label = QLabel("状态: 待机")
        self.status_label.setObjectName("statusLabel")
        top_bar.addWidget(self.title_label)
        top_bar.addStretch()
        top_bar.addWidget(self.status_label)
        cam_layout.addLayout(top_bar)

        self.cam = QLabel()
        self.cam.setObjectName("camView")
        self.cam.setAlignment(Qt.AlignCenter)
        self.cam.setMinimumSize(640, 480)
        self.cam.setText("摄像头未启动")
        cam_layout.addWidget(self.cam, stretch=1)

        main_layout.addWidget(cam_card, stretch=3)

        # --- 右侧：控制面板与身份信息卡片 ---
        right_panel = QVBoxLayout()

        # 1. 功能按钮区
        control_card = QFrame()
        control_card.setObjectName("cardFrame")
        control_layout = QVBoxLayout(control_card)

        self.openButton = QPushButton("📷 打开 / 关闭摄像头")
        self.saveButton = QPushButton("💾 录入人脸与信息")
        self.recogButton = QPushButton("🔍 启动 / 再次识别")
        self.recogButton.setObjectName("btnRecog")
        self.jianceButton = QPushButton("🛡️ 开启活体检测")
        self.jianceButton.setObjectName("btnLive")

        control_layout.addWidget(self.openButton)
        control_layout.addWidget(self.saveButton)
        control_layout.addWidget(self.recogButton)
        control_layout.addWidget(self.jianceButton)
        right_panel.addWidget(control_card)

        # 2. 身份卡片显示区（展示识别结果）
        self.profile_card = QFrame()
        self.profile_card.setObjectName("profileCard")
        profile_layout = QVBoxLayout(self.profile_card)
        profile_layout.setContentsMargins(15, 15, 15, 15)

        card_title = QLabel("=== 身份匹配卡片 ===")
        card_title.setStyleSheet("color: #00F0FF; font-weight: bold; font-size: 14px;")
        card_title.setAlignment(Qt.AlignCenter)
        profile_layout.addWidget(card_title)

        # 姓名
        profile_layout.addWidget(QLabel("姓名", objectName="infoTitle"))
        self.lbl_name = QLabel("未识别", objectName="infoName")
        profile_layout.addWidget(self.lbl_name)

        # 编号
        profile_layout.addWidget(QLabel("人员编号 (ID)", objectName="infoTitle"))
        self.lbl_id = QLabel("--", objectName="infoValue")
        profile_layout.addWidget(self.lbl_id)

        # 年龄
        profile_layout.addWidget(QLabel("年龄", objectName="infoTitle"))
        self.lbl_age = QLabel("--", objectName="infoValue")
        profile_layout.addWidget(self.lbl_age)

        # 单位
        profile_layout.addWidget(QLabel("工作单位 / 部门", objectName="infoTitle"))
        self.lbl_unit = QLabel("--", objectName="infoValue")
        profile_layout.addWidget(self.lbl_unit)

        profile_layout.addStretch()
        right_panel.addWidget(self.profile_card, stretch=1)

        main_layout.addLayout(right_panel, stretch=1)

        # 绑定按钮事件
        self.openButton.clicked.connect(self.on_open_click)
        self.saveButton.clicked.connect(self.on_save_click)
        self.recogButton.clicked.connect(self.on_recog_click)
        self.jianceButton.clicked.connect(self.on_jiance_click)

    def set_status(self, text, color="#00F0FF"):
        self.status_label.setText(f"状态: {text}")
        self.status_label.setStyleSheet(f"color: {color}; background-color: #1F2937; border-radius: 6px; padding: 5px 10px;")

    def update_profile_card(self, user_id=None):
        """刷新右侧身份信息卡片"""
        if user_id is None:
            self.lbl_name.setText("未识别 / 未匹配")
            self.lbl_id.setText("--")
            self.lbl_age.setText("--")
            self.lbl_unit.setText("--")
            return

        users = load_user_data()
        user_info = users.get(str(user_id))
        if user_info:
            self.lbl_name.setText(user_info.get("name", "未知"))
            self.lbl_id.setText(str(user_id))
            self.lbl_age.setText(f"{user_info.get('age', '--')} 岁")
            self.lbl_unit.setText(user_info.get("unit", "未知"))
        else:
            self.lbl_name.setText("未预录入档案")
            self.lbl_id.setText(str(user_id))
            self.lbl_age.setText("--")
            self.lbl_unit.setText("--")

    # 绘制科技感 UI 特效（人脸四角框 + 动态扫描线）
    def draw_tech_effects(self, frame, x, y, w, h, is_matched=True):
        global scan_line_y
        color = (0, 255, 0) if is_matched else (0, 0, 255)
        
        # 1. 四角边框特效
        length = int(w * 0.2)
        thickness = 2
        # 左上角
        cv.line(frame, (x, y), (x + length, y), color, thickness)
        cv.line(frame, (x, y), (x, y + length), color, thickness)
        # 右上角
        cv.line(frame, (x + w, y), (x + w - length, y), color, thickness)
        cv.line(frame, (x + w, y), (x + w, y + length), color, thickness)
        # 左下角
        cv.line(frame, (x, y + h), (x + length, y + h), color, thickness)
        cv.line(frame, (x, y + h), (x, y + h - length), color, thickness)
        # 右下角
        cv.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)

        # 2. 动态扫描线特效
        scan_line_y = (scan_line_y + 4) % h
        line_current_y = y + scan_line_y
        cv.line(frame, (x, line_current_y), (x + w, line_current_y), (255, 240, 0), 1)

    # ---------------- 业务逻辑 ----------------
    def on_open_click(self):
        global cap, timer
        if cap is None:
            self.is_recognized = False
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头！")
                cap = None
                return
            timer = QTimer()
            timer.timeout.connect(self.update_frame)
            timer.start(15)
            self.set_status("普通监控中", "#10B981")
        else:
            self.stop_all_timers()
            self.cam.clear()
            self.cam.setText("摄像头已关闭")
            self.set_status("待机")
            self.update_profile_card(None)

    def update_frame(self):
        global cap
        if cap is not None and cap.isOpened():
            res, frame = cap.read()
            if not res: return
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = facecascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                self.draw_tech_effects(frame, x, y, w, h, is_matched=True)

            self.display_image(frame)

    def update_frame2(self):
        global cap, recognizer, timer

        # 如果已经通过识别并锁定了结果，不再进行画面更新
        if self.is_recognized:
            return

        if cap is not None and cap.isOpened():
            res, frame = cap.read()
            if not res: return
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = facecascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                try:
                    user_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    if conf > 80: # 相似度未达标
                        label = "Scanning / Unknown"
                        self.draw_tech_effects(frame, x, y, w, h, is_matched=False)
                        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else: # 匹配成功！
                        label = f"MATCHED [ID:{user_id}]"
                        self.draw_tech_effects(frame, x, y, w, h, is_matched=True)
                        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 1. 刷新画面与右侧卡片信息
                        self.display_image(frame)
                        self.update_profile_card(user_id)
                        
                        # 2. 设置识别完成标志位，停止画面刷新
                        self.is_recognized = True
                        if timer is not None:
                            timer.stop()

                        self.set_status("识别验证成功！", "#10B981")

                        # 3. 得到信息后弹窗恭喜提示
                        users = load_user_data()
                        user_info = users.get(str(user_id), {})
                        name = user_info.get("name", f"ID:{user_id}")
                        
                        QMessageBox.information(self, "识别提示", f"🎉 恭喜！人脸识别成功！\n\n欢迎您：{name}")
                        return
                except Exception as e:
                    pass

            self.display_image(frame)

    def on_save_click(self):
        """录入人脸并弹窗填写信息"""
        global cap
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = facecascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    roi_gray = gray[y:y+h, x:x+w]

                    # 弹出信息填写窗口
                    dialog = UserInfoDialog(self)
                    if dialog.exec() == QDialog.Accepted:
                        user_id, name, age, unit = dialog.get_data()

                        if not user_id.isdigit():
                            QMessageBox.warning(self, "提示", "编号(ID)必须为纯数字！")
                            return
                        if not name:
                            QMessageBox.warning(self, "提示", "姓名不能为空！")
                            return

                        # 1. 保存图片
                        save_dir = "./data/"
                        os.makedirs(save_dir, exist_ok=True)
                        cv.imwrite(f'{save_dir}/{user_id}.jpg', roi_gray)

                        # 2. 保存完整身份 JSON
                        save_user_info(user_id, name, age, unit)

                        QMessageBox.information(self, "成功", f"人员【{name}】信息档案录入成功！")
                else:
                    QMessageBox.warning(self, "失败", "未识别到人脸，请面向镜头！")

    def on_recog_click(self):
        """启动或再次识别"""
        global cap, timer, recognizer
        if not os.path.exists("./data") or len(os.listdir("./data")) == 0:
            QMessageBox.warning(self, "警告", "人脸库为空，请先录入！")
            return

        # 重置识别完成标志位（解锁画面，清空卡片，开始新一轮识别）
        self.is_recognized = False
        self.update_profile_card(None)

        self.set_status("AI分析与模型加载中...", "#3B82F6")
        QApplication.processEvents()

        try:
            self.faceTrain("./data")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败: {str(e)}")
            return

        self.stop_all_timers()
        if cap is None or not cap.isOpened():
            cap = cv.VideoCapture(0)

        timer = QTimer()
        timer.timeout.connect(self.update_frame2)
        timer.start(15)
        self.set_status("实时比对与检索中...", "#10B981")

    def update_live_detect(self):
        global cap, blink_count, eye_closed_flag, is_live_detect, live_start_time, live_timeout
        if cap is None or not cap.isOpened(): return

        current_time = time.time()
        pass_time = current_time - live_start_time

        if pass_time > live_timeout:
            QMessageBox.warning(self, "活体检测失败", "超时未眨眼，判定为照片攻击！")
            self.on_jiance_click()
            return

        ret, frame = cap.read()
        if not ret: return

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

        tip_text = f"Live Check: {blink_count}/{blink_need} | Timeout: {live_timeout - pass_time:.1f}s"
        cv.putText(frame, tip_text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                self.draw_tech_effects(frame, x, y, w, h, is_matched=True)
                face_roi = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

                if len(eyes) == 0:
                    eye_closed_flag = True
                elif len(eyes) >= 2 and eye_closed_flag:
                    blink_count += 1
                    eye_closed_flag = False

                if blink_count >= blink_need:
                    self.display_image(frame)
                    QMessageBox.information(self, "活体检测", "真人验证通过！")
                    self.on_jiance_click()
                    return

        self.display_image(frame)

    def on_jiance_click(self):
        global cap, timer, live_timer, is_live_detect, blink_count, eye_closed_flag, live_start_time
        self.is_recognized = False
        if not is_live_detect:
            self.stop_all_timers()
            blink_count = 0
            eye_closed_flag = False
            is_live_detect = True
            live_start_time = time.time()

            if cap is None or not cap.isOpened():
                cap = cv.VideoCapture(0)

            live_timer = QTimer()
            live_timer.timeout.connect(self.update_live_detect)
            live_timer.start(15)
            self.set_status("活体防护检测中...", "#F59E0B")
        else:
            is_live_detect = False
            self.stop_all_timers()
            self.cam.clear()
            self.cam.setText("检测已停用")
            self.set_status("待机")

    def faceTrain(self, path):
        Images, labels = [], []
        filenames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        for filename in filenames:
            img = cv.imread(filename)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = facecascade.detectMultiScale(gray)
            try:
                user_id = int(os.path.split(filename)[1].split(".")[0])
                for (x, y, w, h) in faces:
                    Images.append(gray[y:y+h, x:x+w])
                    labels.append(user_id)
            except ValueError:
                continue

        if len(Images) > 0:
            recognizer.train(Images, np.array(labels))
            os.makedirs("./model", exist_ok=True)
            recognizer.write("./model/trainer.yml")

    def display_image(self, frame):
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.cam.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.cam.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def stop_all_timers(self):
        global timer, live_timer, cap, is_live_detect
        is_live_detect = False
        if timer is not None:
            timer.stop()
            timer = None
        if live_timer is not None:
            live_timer.stop()
            live_timer = None
        if cap is not None:
            cap.release()
            cap = None

    def closeEvent(self, event):
        self.stop_all_timers()
        event.accept()

# ==================== 程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionUI()
    window.show()
    sys.exit(app.exec())