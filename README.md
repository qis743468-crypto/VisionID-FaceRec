# 🛡️ CyberFace Recognition System (高阶AI人脸识别与身份比对系统)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-green)
![OpenCV](https://img.shields.io/badge/OpenCV-LBPH-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

一套基于 **PySide6** 与 **OpenCV** 构建的高颜值、暗黑科技风人脸识别与身份档案管理系统。系统集成了动态科技扫描特效、多维度个人档案录入、LBPH 算法特征匹配以及眨眼活体防御检测等核心功能。

---

## ✨ 项目亮点

- 🎨 **暗黑科技风 GUI**：采用自定义 QSS 样式表打造现代化 Dark-Sci-Fi UI 界面，带来极佳的沉浸式体验。
- 🔍 **动态科技扫描**：人脸识别过程中具备**四角瞄准框**与**自上而下的动态光束扫描线**，大幅提升视觉交互感。
- 💳 **身份卡片实时响应**：识别成功后自动锁定画面，并在右侧 HUD 仪表盘实时渲染出被识别人的姓名、ID、年龄与工作单位。
- 🛡️ **防照片/伪造攻击**：基于 Haar 级联特征与眨眼频率算法，提供轻量级的活体防护检测机制。
- 📂 **轻量化数据存储**：无需配置复杂的 SQL 数据库，内置 JSON 方案即可快速完成“人脸特征 - 个人档案”的映射与持久化。

---

## 🛠️ 技术栈

* **GUI 框架**：[PySide6](https://pypi.org/project/PySide6/) (Qt for Python 6)
* **计算机视觉**：[OpenCV-Python](https://pypi.org/project/opencv-python/) / [OpenCV-Contrib-Python](https://pypi.org/project/opencv-contrib-python/)
* **算法模型**：
  * **人脸/眼睛检测**：Haar Cascade Classifiers
  * **人脸识别训练**：LBPH (Local Binary Patterns Histograms)
* **数据格式**：JSON / YAML / Image Arrays

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone [https://github.com/your-username/CyberFace-Recognition-System.git](https://github.com/your-username/CyberFace-Recognition-System.git)
cd CyberFace-Recognition-System
```

### 2. 安装依赖库

建议在 Python 虚拟环境下运行：

```bash
pip install -r requirements.txt
```

> **提示**：如果在国内下载较慢，可使用清华镜像源：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. 配置 Haar 级联分类器

确保 `haar/` 文件夹下存放有以下两个分类器文件：
- `haarcascade_frontalface_alt2.xml`
- `haarcascade_eye_tree_eyeglasses.xml`

### 4. 运行程序

```bash
python main.py
```

---

## 📖 使用指南

1. **📷 打开摄像头**：点击“打开/关闭摄像头”启动画面预览。
2. **💾 录入人脸与信息**：
   - 保证面部在镜头中央，点击“录入人脸与信息”。
   - 在弹出的档案窗口中输入 **数字ID、姓名、年龄、工作单位** 并保存。
3. **🔍 启动 / 再次识别**：
   - 点击“启动/再次识别”，系统将自动根据当前数据训练 LBPH 模型。
   - 识别成功后画面自动定格，右侧卡片亮起并展示档案信息，同时弹窗提示恭喜成功。
   - 点击“再次识别”可重新解锁画面并继续识别新目标。
4. **🛡️ 开启活体检测**：
   - 点击“开启活体检测”，在指定时间内跟随提示完成 2 次眨眼即可通过真人验证。
   - ![界面截图](<img width="1653" height="1067" alt="2dad47a3ce37b9217294ee9b99fa03b5" src="https://github.com/user-attachments/assets/561c402f-f726-4d6a-94ea-e726145e5747" />)
   - ![界面截图](<img width="1653" height="1067" alt="667f5d6fd93c56febcd59d5441846110" src="https://github.com/user-attachments/assets/6dd5b8cf-a3cc-44c5-b3f7-c4b399969d3d" />)
   - ![界面截图](<img width="1653" height="1067" alt="8f378ecc8fb38d12c288c2e8752c7cef" src="https://github.com/user-attachments/assets/36342058-d618-4bf0-a98e-ab083ff78c84" />)
   - ![界面截图](<img width="1653" height="1067" alt="0f96e7d51c66d17fdaaec3ce89d7df0e" src="https://github.com/user-attachments/assets/dae073d1-2d26-435a-81ec-9437a781b0be" />)


---

## 🤝 贡献与反馈

欢迎提交 Issue 或 Pull Request！如果你觉得这个项目对你有帮助，不妨点个 **⭐ Star** 鼓励一下！

---

## 📜 许可证

本项目基于 [MIT License](LICENSE) 开源许可。
