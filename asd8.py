import streamlit as st
import torch
import os
import urllib.request
from ultralytics.yolo.engine.model import YOLO  # ✅ استخدام `YOLO()` بدل `attempt_load()`
from PIL import Image
from datetime import datetime
import pandas as pd
import time

# ✅ إعداد الصفحة
st.set_page_config(page_title="Fire Detection Monitoring", page_icon="🔥", layout="wide")

# ✅ تثبيت المكتبات المطلوبة إذا لم تكن موجودة
os.system("pip install --upgrade torch torchvision opencv-python pandas ultralytics")

# ✅ تحميل النموذج (`best.pt`) من GitHub إذا لم يكن موجودًا
model_dir = "models"
model_filename = "best.pt"
model_path = os.path.join(model_dir, model_filename)
model_url = "https://raw.githubusercontent.com/msj78598/Fire-Detection-Monitoring-System/main/best.pt"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ✅ التأكد من أن `best.pt` لم يتلف أثناء التحميل
if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
    print("❌ ملف best.pt غير موجود أو تالف، سيتم إعادة تحميله...")
    os.remove(model_path) if os.path.exists(model_path) else None
    urllib.request.urlretrieve(model_url, model_path)
    print("✅ تم تحميل best.pt بنجاح!")

# ✅ تحميل النموذج باستخدام `YOLO()`
st.session_state.model = YOLO(model_path)
print("✅ تم تحميل نموذج YOLOv5 بنجاح!")

# ✅ الشريط الجانبي للإعدادات
st.sidebar.title("⚙️ الإعدادات")
st.sidebar.subheader("📊 إصدار تقرير")

# 📅 تحديد الفترة الزمنية لاستخراج التقرير
start_date = st.sidebar.date_input("📅 تاريخ البداية")
end_date = st.sidebar.date_input("📅 تاريخ النهاية")

# زر استخراج التقرير
if st.sidebar.button("استخراج التقرير"):
    if "fire_detections" in st.session_state and st.session_state.fire_detections:
        filtered_detections = [
            detection for detection in st.session_state.fire_detections
            if start_date <= datetime.strptime(detection['time'], "%Y-%m-%d %H:%M:%S").date() <= end_date
        ]
        
        if filtered_detections:
            df = pd.DataFrame(filtered_detections)
            df['image_link'] = df['image'].apply(lambda x: f'=HYPERLINK("{x}", "عرض الصورة")')

            excel_file = "fire_detections_report.xlsx"
            df.to_excel(excel_file, index=False)

            with open(excel_file, "rb") as file:
                st.sidebar.download_button(
                    label="📥 تحميل التقرير",
                    data=file,
                    file_name=excel_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.sidebar.error("❌ لا توجد اكتشافات في الفترة المحددة.")
    else:
        st.sidebar.error("❌ لا توجد اكتشافات لاستخراج التقرير.")

# ✅ **نظام اكتشاف الحرائق**
st.title("🔥 Fire Detection Monitoring System")
st.markdown("<h4 style='text-align: center; color: #FF5733;'>نظام مراقبة لاكتشاف الحريق</h4>", unsafe_allow_html=True)

# قائمة التخزين
if "fire_detections" not in st.session_state:
    st.session_state.fire_detections = []
if "fire_images" not in st.session_state:
    st.session_state.fire_images = []

# زر بدء الكشف
start_detection = st.button('🚨 ابدأ الكشف عن الحريق 🚨')

# **أعلى شاشة المراقبة**
alert_box = st.empty()
stframe = st.empty()  # شاشة الفيديو
fire_images_placeholder = st.empty()

# ✅ **تشغيل الكاميرا واكتشاف الحريق**
if start_detection:
    cap = cv2.VideoCapture(0)
    fire_classes = [0, 1, 2, 3, 4]  # تحديد فئات الحريق من YOLO
    conf_threshold = 0.5  # الحد الأدنى للثقة

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ خطأ في فتح الكاميرا")
            break

        results = st.session_state.model(frame)
        detections = results.pandas().xyxy[0]
        detections = detections[detections['confidence'] > conf_threshold]

        fire_detected = False
        for _, detection in detections.iterrows():
            if detection['class'] in fire_classes:
                fire_detected = True
                x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
                confidence = detection['confidence'] * 100

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"🔥 Fire: {confidence:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                image_filename = f"fire_detected_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(image_filename, frame)

                st.session_state.fire_images.insert(0, {'image': image_filename, 'timestamp': timestamp})
                st.session_state.fire_detections.insert(0, {'time': timestamp, 'image': image_filename, 'confidence': confidence})

    cap.release()
