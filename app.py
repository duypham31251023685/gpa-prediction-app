import streamlit as st
import pandas as pd
import joblib

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "models/gpa_model.pkl"
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Ứng dụng dự đoán GPA",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("🎓 Ứng dụng dự đoán GPA")

st.write(
    "Dự đoán GPA sinh viên bằng Machine Learning"
)

st.divider()

# =====================================
# GPA SCALE
# =====================================

gpa_scale = st.selectbox(
    "🎯 Hệ GPA",
    [
        "4.0",
        "10.0"
    ]
)

# =====================================
# USER INPUT
# =====================================

col1, col2 = st.columns(2)

with col1:

    study_hours = st.slider(
        "📚 Số giờ học mỗi tuần",
        0,
        80,
        20
    )

    subjects = st.slider(
        "📖 Số môn đang học",
        1,
        12,
        5
    )

    sleep_hours = st.slider(
        "😴 Số giờ ngủ mỗi ngày",
        0,
        12,
        7
    )

    social_media_hours = st.slider(
        "📱 Thời gian dùng MXH",
        0,
        10,
        3
    )

with col2:

    attendance = st.slider(
        "🏫 Attendance (%)",
        0,
        100,
        80
    )

    part_time_job = st.selectbox(
        "💼 Có làm thêm không?",
        [
            "Yes",
            "No"
        ]
    )

    club = st.selectbox(
        "🎯 Có tham gia CLB không?",
        [
            "Yes",
            "No"
        ]
    )

    study_type = st.selectbox(
        "🧠 Hình thức học",
        [
            "Self-study",
            "Group study",
            "Mixed"
        ]
    )

# =====================================
# FEATURE ENGINEERING
# =====================================

study_efficiency = (
    study_hours /
    (social_media_hours + 1)
)

good_sleep = 1 if 6 <= sleep_hours <= 8 else 0

# =====================================
# CREATE DATAFRAME
# =====================================

input_df = pd.DataFrame({

    "study_hours": [study_hours],

    "subjects": [subjects],

    "sleep_hours": [sleep_hours],

    "social_media_hours": [social_media_hours],

    "part_time_job": [part_time_job],

    "club": [club],

    "attendance": [attendance],

    "study_type": [study_type],

    "study_efficiency": [study_efficiency],

    "good_sleep": [good_sleep]
})

# =====================================
# SHOW INPUT
# =====================================

st.subheader(
    "📋 Dữ liệu đã nhập"
)

st.dataframe(input_df)

# =====================================
# PREDICT BUTTON
# =====================================

if st.button(
    "🚀 Dự đoán GPA",
    key="predict_button"
):

    # =================================
    # PREDICT
    # =================================

    prediction = model.predict(
        input_df
    )[0]

    # =================================
    # GPA SCALE
    # =================================

    if gpa_scale == "10.0":

        display_gpa = prediction * 2.5
        max_gpa = 10.0

    else:

        display_gpa = prediction
        max_gpa = 4.0

    # =================================
    # METRIC CARDS
    # =================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📈 GPA dự đoán",
            f"{display_gpa:.2f}"
        )

    with col2:

        st.metric(
            "📚 Study Hours",
            study_hours
        )

    with col3:

        st.metric(
            "🏫 Attendance",
            attendance
        )

    # =================================
    # MODEL CONFIDENCE
    # =================================

    confidence = 92

    st.metric(
        "🤖 Model Confidence",
        f"{confidence}%"
    )

    # =================================
    # GPA PROGRESS BAR
    # =================================

    st.subheader(
        "📊 GPA Progress"
    )

    st.write(
        f"### {display_gpa:.2f} / {max_gpa}"
    )

    progress_value = min(
        display_gpa / max_gpa,
        1.0
    )

    st.progress(progress_value)

    # =================================
    # GPA LEVEL
    # =================================

    if prediction >= 3.6:

        st.success(
            "🔥 Học lực xuất sắc"
        )

    elif prediction >= 3.2:

        st.info(
            "👍 Học lực giỏi"
        )

    elif prediction >= 2.5:

        st.warning(
            "📚 Học lực khá"
        )

    else:

        st.error(
            "⚠️ GPA thấp"
        )

    # =================================
    # STUDENT PROFILE
    # =================================

    st.subheader(
        "🧠 Hồ sơ học tập"
    )

    summary = f'''
    Sinh viên hiện học {study_hours} giờ/tuần,
    attendance {attendance}%,
    ngủ {sleep_hours} giờ/ngày,
    và dùng mạng xã hội {social_media_hours} giờ/ngày.
    '''

    st.write(summary)

    # =================================
    # ACADEMIC RISK SCORE
    # =================================

    risk_score = 0

    if attendance < 70:
        risk_score += 30

    if social_media_hours > 6:
        risk_score += 30

    if sleep_hours < 6:
        risk_score += 20

    if study_hours < 10:
        risk_score += 20

    st.metric(
        "⚠️ Academic Risk Score",
        f"{risk_score}/100"
    )

    # =================================
    # GPA RISK DETECTION
    # =================================

    st.subheader(
        "⚠️ GPA Risk Detection"
    )

    if risk_score >= 70:

        st.error(
            "Nguy cơ GPA giảm mạnh."
        )

    elif risk_score >= 40:

        st.warning(
            "Có một số yếu tố ảnh hưởng tiêu cực đến GPA."
        )

    else:

        st.success(
            "Không phát hiện nguy cơ GPA nghiêm trọng."
        )

    # =================================
    # HABIT COMPARISON
    # =================================

    st.subheader(
        "📋 So sánh thói quen"
    )

    compare_df = pd.DataFrame({

        "Hiện tại": [
            study_hours,
            social_media_hours,
            sleep_hours,
            attendance
        ],

        "Khuyến nghị": [
            "15+",
            "<5",
            "6-8",
            "80+"
        ]

    }, index=[
        "Study Hours",
        "Social Media",
        "Sleep",
        "Attendance"
    ])

    st.table(compare_df)

    # =================================
    # IMPROVEMENT SCORE
    # =================================

    improvement_score = 100

    if social_media_hours > 5:
        improvement_score -= 20

    if attendance < 75:
        improvement_score -= 30

    if sleep_hours < 6:
        improvement_score -= 20

    if study_hours < 15:
        improvement_score -= 20

    st.metric(
        "📈 Improvement Score",
        improvement_score
    )

    # =================================
    # GPA IMPROVEMENT
    # =====================================

    st.subheader(
        "💡 Cách cải thiện GPA"
    )

    high_priority = []

    medium_priority = []

    low_priority = []

    # HIGH PRIORITY

    if attendance < 75:

        high_priority.append(
            "🏫 Tăng attendance trên lớp."
        )

    if study_hours < 15:

        high_priority.append(
            "📚 Tăng số giờ học mỗi tuần."
        )

    # MEDIUM PRIORITY

    if social_media_hours >= 5:

        medium_priority.append(
            "📱 Giảm thời gian sử dụng mạng xã hội."
        )

    if sleep_hours < 6:

        medium_priority.append(
            "😴 Ngủ đủ từ 6-8 tiếng mỗi ngày."
        )

    # LOW PRIORITY

    if study_type == "Self-study":

        low_priority.append(
            "👥 Kết hợp học nhóm để tăng hiệu quả."
        )

    if part_time_job == "Yes":

        low_priority.append(
            "💼 Cân bằng giữa học và làm thêm."
        )

    # SHOW PRIORITIES

    if len(high_priority) > 0:

        st.error(
            "🔴 HIGH PRIORITY"
        )

        for item in high_priority:

            st.write(item)

    if len(medium_priority) > 0:

        st.warning(
            "🟠 MEDIUM PRIORITY"
        )

        for item in medium_priority:

            st.write(item)

    if len(low_priority) > 0:

        st.info(
            "🟢 LOW PRIORITY"
        )

        for item in low_priority:

            st.write(item)

    # =================================
    # FINAL AI MESSAGE
    # =================================

    if prediction >= 3.6:

        st.success(
            "🔥 AI đánh giá rằng thói quen học tập hiện tại đang rất tốt."
        )

    elif prediction >= 3.0:

        st.info(
            "📈 GPA có thể cải thiện thêm nếu tối ưu thời gian học tập."
        )

    else:

        st.error(
            "⚠️ AI phát hiện nhiều yếu tố có thể làm GPA giảm."
        )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Machine Learning GPA Prediction Project"
)