# streamlit run app.py
import torch

from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])


import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import time

import io

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detections')
        
        worksheet = writer.sheets['Detections']
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_len
            
    return output.getvalue()

st.set_page_config(page_title="AI Weapon Detector", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('best.pt') 

model = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0

st.title("Мониторинг безопасности в реальном времени")

conf_threshold = st.sidebar.slider("Порог уверенности", 0.1, 1.0, 0.5)
target_classes = st.sidebar.multiselect("Отслеживать классы:", 
                                        list(model.names.values()), 
                                        default=list(model.names.values()))

run_stream = st.sidebar.checkbox("Запустить трансляцию")

st.sidebar.markdown("---")
st.sidebar.subheader("Экспорт данных")

if st.session_state.history:
    report_df = pd.DataFrame(st.session_state.history)
    
    excel_data = convert_df_to_excel(report_df)
    st.sidebar.download_button(
        label="📥 Скачать отчет в Excel",
        data=excel_data,
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    if st.sidebar.button("🗑️ Очистить историю"):
        st.session_state.history = []
        st.session_state.last_detection_time = 0
        st.rerun()
else:
    st.sidebar.info("История событий пуста")

col1, col2 = st.columns([3, 1])
frame_placeholder = col1.empty()
history_placeholder = col2.empty()

if run_stream:
    cap = cv2.VideoCapture(0) 
    
    while cap.isOpened() and run_stream:
        success, frame = cap.read()
        if not success:
            st.error("Не удалось получить доступ к камере.")
            break

        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        found_any_target = False
        current_time = time.time()
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if class_name in target_classes:
                found_any_target = True
                if current_time - st.session_state.last_detection_time > 2:
                    now = datetime.now().strftime("%H:%M:%S")
                    st.session_state.history.append({
                        "Время": now,
                        "Объект": class_name,
                        "Уверенность": round(float(box.conf[0]), 2)
                    })
                    st.session_state.last_detection_time = current_time

        frame_placeholder.image(annotated_frame, channels="RGB", width='stretch')
        
        with history_placeholder.container():
            st.subheader("События")
            if st.session_state.history:
                df = pd.DataFrame(st.session_state.history).iloc[::-1]
                st.table(df.head(15))

        if not run_stream:
            break
            
    cap.release()
else:
    st.info("Нажмите 'Запустить трансляцию' в боковой панели")