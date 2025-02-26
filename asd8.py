import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import torch
import urllib.request
import cv2
import numpy as np
import time
from PIL import Image
from datetime import datetime

# ✅ تثبيت المتطلبات تلقائيًا عند الحاجة (خيار إضافي)
os.system("pip install --upgrade ultralytics opencv-python-headless streamlit-webrtc")

# ✅ تحميل نموذج YOLOv5
MODEL_PATH = "best.pt"
MODEL_URL = "https://raw.githubusercontent.com/msj78598/Fire-Detection-Monitoring-System/main/best.pt"

# ✅ التحقق من وجود النموذج وتحميله إن لم يكن موجودًا
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
    st.warning("📥 يتم تحميل النموذج... يرجى الانتظار!")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("✅ تم تحميل النموذج بنجاح!")

# ✅ تحميل YOLOv5 وتخزينه في `session_state` لضمان استمراريته
if "model" not in st.session_state:
    try:
        st.session_state.model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, source="github")
        st.success("✅ تم تحميل نموذج YOLOv5 بنجاح!")
    except Exception as e:
        st.error(f"❌ خطأ في تحميل YOLOv5: {e}")

# ✅ إعداد صفحة التطبيق
st.set_page_config(page_title="Fire Detection Monitoring", page_icon="🔥", layout="wide")
st.title("🔥 Fire Detection Monitoring System")
st.markdown("<h4 style='text-align: center; color: #FF5733;'>نظام مراقبة لاكتشاف الحريق</h4>", unsafe_allow_html=True)

# ✅ إضافة خيار الإدخال (كاميرا مباشرة أو رفع صورة/فيديو)
mode = st.sidebar.radio("📌 اختر طريقة الإدخال:", ["🎥 الكاميرا المباشرة", "📂 رفع صورة أو فيديو"])

# 🔥 **1️⃣ تشغيل الكاميرا المباشرة عبر Streamlit WebRTC**
if mode == "🎥 الكاميرا المباشرة":
    st.sidebar.warning("⚠️ تأكد من السماح للمتصفح بالوصول إلى الكاميرا!")

    class FireDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # 🔹 تشغيل النموذج على الصورة
            results = st.session_state.model(img)

            # 🔹 رسم المربعات على الصورة عند اكتشاف الحريق
            fire_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.1:  # العتبة عند 0.1
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "🔥 Fire Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    fire_detected = True

            # 🔴 تشغيل الإنذار عند اكتشاف الحريق
            if fire_detected:
                st.warning("🚨🔥 إنذار! تم اكتشاف حريق!")
                for _ in range(3):
                    st.markdown("<div style='background-color: red; color: white; font-size: 24px; text-align: center;'>🚨🔥 إنذار حريق! 🔥🚨</div>", unsafe_allow_html=True)
                    time.sleep(0.5)
                    st.markdown("<div style='background-color: white; font-size: 24px; text-align: center;'> </div>", unsafe_allow_html=True)
                    time.sleep(0.5)
                
                # تشغيل الإنذار الصوتي
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", format="audio/wav")

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="fire-detection", video_transformer_factory=FireDetectionTransformer)

# 📂 **2️⃣ تحليل صورة أو فيديو مرفوع**
elif mode == "📂 رفع صورة أو فيديو":
    uploaded_file = st.sidebar.file_uploader("📸 قم برفع صورة أو فيديو", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file:
        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            # ✅ تحليل الصورة
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # 🔹 تشغيل YOLOv5 على الصورة
            results = st.session_state.model(image_np)

            # 🔹 رسم المربعات على الصورة عند اكتشاف الحريق
            fire_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.1:  # العتبة عند 0.1
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    fire_detected = True

            # ✅ عرض النتائج
            st.image(image_np, caption="🔍 نتيجة تحليل الصورة", use_column_width=True)

            if fire_detected:
                st.warning("🚨🔥 تم اكتشاف حريق في الصورة!")
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", format="audio/wav")

        elif file_type == "video":
            # ✅ تشغيل الفيديو وتحليله إطار بإطار
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 🔹 تشغيل YOLOv5 على الإطار الحالي
                results = st.session_state.model(frame)

                # 🔹 رسم المربعات على الصورة عند اكتشاف الحريق
                fire_detected = False
                for *xyxy, conf, cls in results.xyxy[0]:
                    if conf > 0.1:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        fire_detected = True

                # ✅ عرض الفيديو بعد التحليل
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, caption="🔍 تحليل الفيديو", use_column_width=True)

            cap.release()

            if fire_detected:
                st.warning("🚨🔥 تم اكتشاف حريق في الفيديو!")
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", format="audio/wav")

st.success("✅ التطبيق جاهز للتشغيل!")
