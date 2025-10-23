# 🐘 Elephant AI Identifier - Hackathon Edition

ระบบ AI แยกช้างด้วยเทคโนโลยี Edge Detection, Pose Estimation และ Machine Learning สำหรับการแข่งขัน Hackathon

## ✨ ฟีเจอร์หลัก

- **🎯 AI Multi-Method Analysis**: ใช้หลายวิธีในการวิเคราะห์รวมกัน
  - Edge Detection สำหรับแยกขอบเขต
  - Pose Estimation สำหรับวิเคราะห์ท่าทาง
  - Image Classification สำหรับจำแนกประเภท
  - Grayscale Analysis สำหรับการประมวลผลภาพ

- **🚀 Real-time Training**: ฝึกสอนโมเดลแบบ real-time
- **📱 User-friendly Interface**: UI/UX ที่สวยงามและใช้งานง่าย
- **🎨 Modern Design**: ดีไซน์ทันสมัยพร้อม Gradient และ Animation
- **📊 Progress Tracking**: แสดงความคืบหน้าและสถิติการฝึกสอน

## 🛠️ เทคโนโลยีที่ใช้

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**: 
  - OpenCV สำหรับ Image Processing
  - MediaPipe สำหรับ Pose Estimation
  - Scikit-learn สำหรับ Machine Learning
  - TensorFlow สำหรับ Deep Learning
- **Computer Vision**:
  - Edge Detection (Canny)
  - Grayscale Conversion
  - Feature Extraction
  - Texture Analysis

## 📦 การติดตั้ง

### 1. Clone Repository
```bash
git clone <https://github.com/NNlnwza/Elephant.git>
cd elephant-ai-identifier
```

### 2. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3. รันแอปพลิเคชัน
```bash
python app.py
```

### 4. เปิดเบราว์เซอร์
ไปที่ `http://localhost:5000`

## 🎯 วิธีการใช้งาน

### 1. ฝึกสอนโมเดล
1. อัปโหลดรูปภาพหรือวิดีโอช้าง
2. ใส่ชื่อช้าง
3. คลิก "อัปโหลดข้อมูลฝึกสอน"
4. ทำซ้ำสำหรับช้างตัวอื่นๆ
5. คลิก "เริ่มเทรนโมเดล"

### 2. ระบุตัวตนช้าง
1. อัปโหลดรูปภาพช้างที่ต้องการระบุ
2. คลิก "ระบุตัวตนช้าง"
3. ดูผลลัพธ์และความแม่นยำ

## 🔬 วิธีการทำงานของ AI

### 1. Feature Extraction
ระบบจะสกัดคุณสมบัติจากภาพโดยใช้หลายวิธี:

- **Edge Detection**: ใช้ Canny Edge Detection เพื่อหาเส้นขอบ
- **Grayscale Analysis**: แปลงภาพเป็น grayscale และวิเคราะห์สถิติ
- **Texture Features**: คำนวณ texture energy และ contrast
- **Shape Features**: วิเคราะห์ความซับซ้อนและอัตราส่วนของรูปร่าง
- **Pose Estimation**: ใช้ MediaPipe วิเคราะห์ท่าทาง

### 2. Machine Learning Pipeline
- ใช้ Random Forest Classifier
- รวมคุณสมบัติจากทุกวิธี
- ฝึกสอนแบบ ensemble learning

### 3. Prediction
- สกัดคุณสมบัติจากภาพใหม่
- ใช้โมเดลที่ฝึกสอนแล้วทำนาย
- แสดงผลลัพธ์พร้อมความมั่นใจ

## 📁 โครงสร้างโปรเจค

```
elephant-ai-identifier/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── templates/
│   └── index.html        # Main UI
├── static/               # Static files
├── uploads/              # Uploaded files
└── models/               # Trained models
```

## 🎨 UI/UX Features

- **Responsive Design**: รองรับทุกขนาดหน้าจอ
- **Drag & Drop**: ลากไฟล์มาวางได้
- **Progress Indicators**: แสดงความคืบหน้า
- **Real-time Feedback**: แสดงสถานะแบบ real-time
- **Modern Animations**: เอฟเฟกต์และ animation ที่สวยงาม

## 🚀 พร้อมสำหรับ Hackathon

### ข้อดีสำหรับการแข่งขัน:
1. **เร็ว**: ใช้งานได้ทันทีไม่ต้องติดตั้งซับซ้อน
2. **ยืดหยุ่น**: รองรับข้อมูลใหม่ได้ทันที
3. **แม่นยำ**: ใช้หลายวิธีวิเคราะห์รวมกัน
4. **สวยงาม**: UI/UX ที่น่าประทับใจ
5. **ครบครัน**: มีทุกฟีเจอร์ที่จำเป็น

### การปรับแต่งสำหรับงานจริง:
- เพิ่มข้อมูลช้างจริงในวันแข่งขัน
- ปรับแต่งพารามิเตอร์ตามข้อมูลจริง
- เพิ่มฟีเจอร์ตามความต้องการ

## 🔧 การปรับแต่ง

### ปรับแต่ง Feature Extraction
แก้ไขใน `app.py` ในฟังก์ชัน `extract_features_from_image()`:

```python
def extract_features_from_image(self, image_path):
    # เพิ่มหรือปรับแต่งการสกัดคุณสมบัติ
    pass
```

### ปรับแต่ง UI
แก้ไขใน `templates/index.html`:

```html
<!-- ปรับแต่งสี, ฟอนต์, หรือเลย์เอาต์ -->
```

## 📊 Performance Tips

1. **ข้อมูลภาพ**: ใช้ภาพที่มีความละเอียดเหมาะสม (ไม่ใหญ่เกินไป)
2. **จำนวนตัวอย่าง**: ควรมีอย่างน้อย 5-10 รูปต่อช้าง
3. **คุณภาพภาพ**: ภาพที่ชัดเจนจะให้ผลลัพธ์ที่ดีกว่า
4. **แสงสว่าง**: ภาพที่มีแสงสว่างสม่ำเสมอจะดีกว่า

## 🐛 Troubleshooting

### ปัญหาที่พบบ่อย:

1. **ไม่สามารถอัปโหลดไฟล์**
   - ตรวจสอบขนาดไฟล์ (ไม่เกิน 100MB)
   - ตรวจสอบประเภทไฟล์ (JPG, PNG, MP4, AVI)

2. **โมเดลไม่แม่นยำ**
   - เพิ่มจำนวนตัวอย่างข้อมูล
   - ใช้ภาพที่มีคุณภาพดี
   - ตรวจสอบการตั้งชื่อช้าง

3. **แอปไม่ทำงาน**
   - ตรวจสอบการติดตั้ง dependencies
   - ตรวจสอบ port 5000 ว่างหรือไม่

## 📞 การสนับสนุน

หากมีปัญหาหรือข้อสงสัย:
1. ตรวจสอบ README นี้ก่อน
2. ดู error messages ใน console
3. ตรวจสอบ logs ของ Flask

## 🎉 พร้อมสำหรับ Hackathon!

ระบบนี้พร้อมใช้งานสำหรับการแข่งขัน Hackathon แล้ว! เพียงแค่:
1. ติดตั้ง dependencies
2. รันแอปพลิเคชัน
3. อัปโหลดข้อมูลช้างจริงในวันแข่งขัน
4. ฝึกสอนโมเดล
5. ใช้งานได้ทันที!

**ขอให้โชคดีในการแข่งขัน! 🚀🐘**

