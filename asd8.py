import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import urllib.request
import cv2
import av
import numpy as np
from PIL import Image
from datetime import datetime
from ultralytics import YOLO

# ✅ تحميل `best.pt` إذا لم يكن موجودًا
MODEL_PATH = "best.pt"
MODEL_URL = "https://raw.githubusercontent.com/msj78598/Fire-Detection-Monitoring-System/main/best.pt"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
    st.warning("🔄 يتم تحميل النموذج... يرجى الانتظار!")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("✅ تم تحميل النموذج بنجاح!")

# ✅ تحميل نموذج YOLO
try:
    model = YOLO(MODEL_PATH)
    st.success("✅ تم تحميل YOLOv5 بنجاح!")
except Exception as e:
    st.error(f"❌ خطأ في تحميل YOLOv5: {e}")
    st.stop()

# ✅ إعداد الصفحة
st.set_page_config(page_title="Fire Detection Monitoring", page_icon="🔥", layout="wide")

# ✅ اختيار الإدخال
mode = st.sidebar.radio("📌 اختر طريقة الإدخال:", ["🎥 الكاميرا المباشرة", "📂 رفع صورة أو فيديو"])

# ✅ تشغيل الكاميرا عبر `Streamlit WebRTC`
if mode == "🎥 الكاميرا المباشرة":
    st.sidebar.warning("⚠️ تأكد من السماح للمتصفح بالوصول إلى الكاميرا.")

    class FireDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 🔹 تشغيل YOLOv5 على الإطار الحالي
            results = model.predict(img_rgb, conf=0.3)

            fire_detected = False
            for r in results:
                for box in r.boxes:
                    if box.conf[0] > 0.3:
                        fire_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, "🔥 Fire Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if fire_detected:
                st.warning("🚨🔥 تم اكتشاف حريق! 🔥🚨")
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", autoplay=True)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="fire-detection", video_transformer_factory=FireDetectionTransformer,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ✅ تحليل صورة أو فيديو
elif mode == "📂 رفع صورة أو فيديو":
    uploaded_file = st.sidebar.file_uploader("📸 قم برفع صورة أو فيديو", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file:
        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # 🔹 تشغيل YOLOv5 على الصورة
            results = model.predict(image_np, conf=0.3)

            # 🔹 رسم المربعات على الصورة
            for r in results:
                for box in r.boxes:
                    if box.conf[0] > 0.3:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

            st.image(image_np, caption="🔍 نتيجة تحليل الصورة", use_column_width=True)
            st.success("✅ تم تحليل الصورة بنجاح!")

        elif file_type == "video":
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=0.3)

                for r in results:
                    for box in r.boxes:
                        if box.conf[0] > 0.3:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, caption="🔍 تحليل الفيديو", use_column_width=True)

            cap.release()
            st.success("✅ تم تحليل الفيديو بنجاح!")
