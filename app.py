import os
os.system('apt-get update && apt-get install -y libgl1-mesa-glx')

# All imports
import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import torch
import time
from datetime import datetime

# Load models
yolo8n = YOLO('yolov8n.pt')
yolo8s = YOLO('yolov8s.pt')
yolo5s = YOLO('yolov5s.pt')

# Page setup
st.set_page_config(page_title="JLL's VisionGuard", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
    .custom-title, .custom-subtitle {
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .custom-title {
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .custom-subtitle {
        font-size: 1.4rem;
        color: #666;
        margin-top: 0;
    }
    [data-testid="stSidebar"] {
        width: 200px !important;
    }
    .sidebar-box {
        padding: 10px 5px 10px 15px;
        background-color: #f4f4f4;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .sidebar-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
        font-size: 1.1rem;
    }
    .center-video {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 class='custom-title'>🛡️ JLL's VisionGuard</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='custom-subtitle'>Real-Time AI Vision Intelligence</h2>", unsafe_allow_html=True)

# Layout
left_col, right_col = st.columns([1.2, 1.8])

# Sidebar in left_col
with left_col:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>🧠 Tool Selector</div>", unsafe_allow_html=True)
    task = st.radio("Choose Task", [
        "🧍‍ Temperature Based on People Count",
        "🎉 Crowd Alert (>3 People)",
        "🎯 Pest Detection",
        "📺 Data Sensitive Streaming"
    ], key="task", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>🎥 Input Source</div>", unsafe_allow_html=True)
    input_type = st.radio("Input Type", [
        "📷 Webcam", "📁 Upload", "🌐 CCTV URL"
    ], key="input_type", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

# Video input logic
def get_video_source():
    if input_type == "📷 Webcam":
        return cv2.VideoCapture(0), True
    elif input_type == "📁 Upload":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            st.session_state['uploaded_file_path'] = tfile.name
            return cv2.VideoCapture(tfile.name), True
    elif input_type == "🌐 CCTV URL":
        url = st.text_input("RTSP/HTTP URL")
        if url:
            return cv2.VideoCapture(url), True
    return None, False

# Writer
def get_writer(filename, frame_shape, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(filename, fourcc, fps, (frame_shape[1], frame_shape[0]))

# ==== TASKS ====


# 1. Temperature Based on People Count
if task == "🧍‍ Temperature Based on People Count":
    def get_temperature(count):
        return max(20, 26 - count)

    def process_frame_temp(frame, results):
        people = [box for box, cls in zip(results.boxes.xyxy, results.boxes.cls) if int(cls) == 0]
        count = len(people)
        temp = get_temperature(count)
        for box in people:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"People: {count} | Temp: {temp}°C"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return frame, count, temp

    with right_col:
        cap, ready = get_video_source()
        if ready and st.button("Start"):
            out_path = "output_temp.avi"
            frame_placeholder = st.empty()
            video_writer = None
            alerts = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                small = cv2.resize(frame, (480, 360))
                results = yolo8n(small, conf=0.4)[0]
                processed, count, temp = process_frame_temp(frame, results)

                if video_writer is None:
                    video_writer = get_writer(out_path, frame.shape)

                video_writer.write(processed)
                st.markdown("<div class='center-video'>", unsafe_allow_html=True)
                frame_placeholder.image(processed, channels="BGR")
                st.markdown("</div>", unsafe_allow_html=True)

            cap.release()
            video_writer.release()

            # Alert prompt for temperature (example if count > 5)
            if count > 5:
                alerts.append(f"⚠️ High people count detected: {count}, Temperature adjusted to {temp}°C")

            if alerts:
                for a in alerts:
                    st.warning(a)

            # Show download button after processing finishes
            if os.path.exists(out_path):
                with open(out_path, "rb") as file:
                    st.download_button("Download Processed Video", file, "temperature_output.avi")


# 2. Crowd Alert
elif task == "🎉 Crowd Alert (>3 People)":
    def detect_crowd(frame):
        results = yolo5s(frame)
        detections = results.xyxy[0].cpu().numpy()
        people = [box for box in detections if int(box[5]) == 0]
        alert_flag = False

        if len(people) > 3:
            alert_flag = True
            # Draw big, bold, red outlined "CROWD ALERT!"
            cv2.putText(frame, "CROWD ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, "CROWD ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 2, cv2.LINE_AA)
        for box in people:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        return frame, alert_flag, len(people)

    with right_col:
        cap, ready = get_video_source()
        if ready and st.button("Start Crowd Monitor"):
            out_path = "output_crowd.avi"
            frame_placeholder = st.empty()
            video_writer = None
            alerts = []
            crowd_alert_shown = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, crowd_alert, people_count = detect_crowd(frame)

                if crowd_alert:
                    if not crowd_alert_shown:
                        st.warning(f"🚨 Crowd Alert! {people_count} people detected.")
                        crowd_alert_shown = True
                    alerts.append(f"Crowd Alert: {people_count} people detected.")

                if video_writer is None:
                    video_writer = get_writer(out_path, frame.shape)

                video_writer.write(frame)
                st.markdown("<div class='center-video'>", unsafe_allow_html=True)
                frame_placeholder.image(frame, channels="BGR")
                st.markdown("</div>", unsafe_allow_html=True)

            cap.release()
            video_writer.release()

            if alerts:
                # Show unique alerts once after video ends
                unique_alerts = list(set(alerts))
                for alert_msg in unique_alerts:
                    st.warning(alert_msg)

            # Download button
            if os.path.exists(out_path):
                with open(out_path, "rb") as file:
                    st.download_button("Download Processed Video", file, "crowd_output.avi")


# 3. Pest Detection
elif task == "🎯 Pest Detection":
    with right_col:
        cap, ready = get_video_source()
        if ready and st.button("Start Motion Detector"):
            out_path = "output_pest.avi"
            ret, bg_frame = cap.read()
            if not ret:
                st.error("Failed to read from video source.")
            else:
                bg_gray = cv2.GaussianBlur(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                stframe = st.empty()
                os.makedirs("screenshots", exist_ok=True)
                last_alert = 0
                video_writer = None
                alerts = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                    delta = cv2.absdiff(bg_gray, gray)
                    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(cv2.dilate(thresh, None, 2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    alert = False
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 800:
                            continue
                        x, y, w, h = cv2.boundingRect(cnt)
                        roi = frame[y:y+h, x:x+w]
                        results = yolo8n(roi)[0]
                        labels = [yolo8n.names[int(box[-1])] for box in results.boxes.data]
                        if 'person' not in labels:
                            alert = True
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if alert and time.time() - last_alert > 10:
                        last_alert = time.time()
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = f"screenshots/movement_{ts}.jpg"
                        cv2.imwrite(path, frame)
                        alert_msg = f"📸 Movement Detected (No Human)! Screenshot saved: {path}"
                        st.warning(alert_msg)
                        alerts.append(alert_msg)

                    if video_writer is None:
                        video_writer = get_writer(out_path, frame.shape)

                    video_writer.write(frame)
                    stframe.image(frame, channels="BGR")

                cap.release()
                video_writer.release()

                if alerts:
                    unique_alerts = list(set(alerts))
                    for alert_msg in unique_alerts:
                        st.warning(alert_msg)

                # Download button
                if os.path.exists(out_path):
                    with open(out_path, "rb") as file:
                        st.download_button("Download Processed Video", file, "pest_output.avi")


# 4. Data Sensitive Streaming - Blur Screens
elif task == "📺 Data Sensitive Streaming":
    SCREEN_CLASSES = [62, 63, 64, 65, 66, 67, 75]

    def blur_screens(frame):
        results = yolo8s.predict(frame, conf=0.1, iou=0.4)[0]
        screen_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in SCREEN_CLASSES and conf > 0.1:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (91, 91), 0)
                    frame[y1:y2, x1:x2] = blurred
                    screen_boxes.append((x1, y1, x2, y2))
        return frame, screen_boxes

    with right_col:
        cap, ready = get_video_source()
        if ready and st.button("Start Blurring"):
            out_path = "output_blur.avi"
            stframe = st.empty()
            video_writer = None
            alerts = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed, screens = blur_screens(frame.copy())

                if screens:
                    alerts.append(f"Blurred {len(screens)} sensitive screen(s) in frame.")

                if video_writer is None:
                    video_writer = get_writer(out_path, processed.shape)

                video_writer.write(processed)
                stframe.image(processed, channels="BGR")

            cap.release()
            video_writer.release()

            if alerts:
                unique_alerts = list(set(alerts))
                for alert_msg in unique_alerts:
                    st.info(alert_msg)

            # Download button
            if os.path.exists(out_path):
                with open(out_path, "rb") as file:
                    st.download_button("Download Processed Video", file, "blurred_output.avi")
