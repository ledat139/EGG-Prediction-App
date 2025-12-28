import streamlit as st
import os
import base64
import pandas as pd
from preprocess.load_eeg import load_eeg
from visualize.visualize import plot_raw_eeg, plot_cwt_grid
from inference.predict import predict_with_voting
import tempfile

CLASS_MAP = {
    "A": "Bệnh Alzheimer",
    "F": "Sa sút trí tuệ thùy trán - thái dương",
    "C": "Người khỏe mạnh"
}
# ================= CONFIG =================
st.set_page_config(layout="wide")

# ================= LOAD LOGO =================
def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = load_logo("assets/uit_is_logo.png")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# ================= CSS =================
st.markdown("""
<style>
/* ===== GLOBAL ===== */
body {
    background-color: #ffffff;
    color: #003366;
}

.block-container {
    padding-top: 0.5rem;
}

/* ===== HEADER ===== */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    padding: 12px 24px;
    margin-top: 35px;
    border-bottom: 4px solid #2f80ed;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.header-left img {
    height: 58px;
}

.header-left-text {
    font-size: 14px;
    line-height: 1.4;
    font-weight: 600;
    color: #003366;
}

.header-right {
    text-align: right;
    font-size: 15px;
    font-weight: 700;
    color: #003366;
    max-width: 45%;
}

/* ===== SECTION ===== */
.section {
    background: #ffffff;
    padding: 18px 20px;
    margin-bottom: 20px;
    border: 1px solid #dbe7f5;
    border-radius: 6px;
}

/* ===== BUTTON ===== */
.stButton > button {
    background-color: #2f80ed;
    color: white;
    border-radius: 6px;
    border: none;
    padding: 8px 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1f4fd8;
}

/* ===== RESULT BOX ===== */
.result-box {
    background-color: #f2f7ff;
    border-left: 6px solid #2f80ed;
    padding: 14px;
    font-size: 17px;
    font-weight: 700;
    color: #003366;
    border-radius: 4px;
}

/* ===== TABLE ===== */
table {
    border-collapse: collapse;
    width: 100%;
}

thead {
    background-color: #eaf2ff;
    color: #003366;
}

td, th {
    border: 1px solid #cfe0f5;
    padding: 6px;
    text-align: center;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background-color: #ffffff;
    border-bottom: 2px solid #2f80ed;
}

.stTabs [data-baseweb="tab"] {
    color: #003366;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #2f80ed;
    border-bottom: 3px solid #2f80ed;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(f"""
<div class="header">
  <div class="header-left">
    <img src="data:image/png;base64,{logo}">
    <div class="header-left-text">
      Trường Đại học Công nghệ Thông tin, ĐHQG-HCM<br>
      <b>Khoa Hệ thống Thông tin</b>
    </div>
  </div>

  <div class="header-right">
    CÔNG CỤ DỰ ĐOÁN BỆNH ALZHEIMER<br>
    SA SÚT TRÍ TUỆ THÙY TRÁN – THÁI DƯƠNG
    VÀ NGƯỜI KHỎE MẠNH
  </div>
</div>
""", unsafe_allow_html=True)

# ================= TABS =================
tab_upload, tab_predict = st.tabs(["Tải dữ liệu", "Dự đoán"])

# =====================================================
# ================= TAB 1: UPLOAD =====================
# =====================================================
with tab_upload:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Hướng dẫn upload dữ liệu")
        st.markdown("""
            Upload file `.set`: File tín hiệu EEG
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Tiến hành upload", type=[".set"])
        if uploaded:
            st.success("Upload thành công")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        # Lấy tên file gốc
        file_name = uploaded.name
        save_path = os.path.join(UPLOAD_DIR, file_name)

        # Lưu file vào thư mục uploads
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Đang xử lý raw EEG..."):
            segments, raw_data, preview_cwt = load_eeg(
                save_path,
                n_jobs=-1   # dùng toàn bộ CPU
            )
        ch_names = raw_data.info["ch_names"]
        st.subheader("EEG thô")
        plot_raw_eeg(raw_data.get_data(), ch_names=ch_names)

        st.subheader("CWT preview (segment đầu)")
        plot_cwt_grid(preview_cwt, ch_names=ch_names)
        st.session_state.segments = segments
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# ================= TAB 2: PREDICT ====================
# =====================================================
with tab_predict:
    st.markdown('<div class="section">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Chọn model")
        model_family = st.radio(
            "Loại mô hình",
            [
                "CNN",
                "Vision Transformer",
                "ResNet-18",
                "EEGConvNeXt",
                "CNN + ViT",
                "ResNet-18 + ViT",
                "EEGConvNeXt + ViT",
            ],
        )

        predict_btn = st.button("Dự đoán")

    with col2:
        if predict_btn:
            # ====== CHECK SEGMENTS ======
            if "segments" not in st.session_state or st.session_state.segments is None:
                st.error("Chưa có dữ liệu EEG. Vui lòng upload và preprocess trước.")
                st.stop()

            segments = st.session_state.segments

            with st.spinner("Đang thực hiện ensemble voting..."):
                result = predict_with_voting(segments, model_family)

            label = result["final_vote"]
            label_name = CLASS_MAP.get(label, label)
            # ================= KẾT QUẢ CUỐI =================
            st.markdown('<div class="big-result">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="result-box" style="text-align:center;">
                    KẾT QUẢ DỰ ĐOÁN CUỐI CÙNG: <b>{label_name}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # ================= SEGMENT-LEVEL =================
            st.subheader("Segment-level prediction per model")

            table_rows = []

            for model_name, info in result["model_votes"].items():
                row = {"Model": model_name}
                for cls in ["A", "F", "C"]:
                    row[cls] = info["segment_counts"].get(cls, 0)
                row["Total segments"] = sum(row[c] for c in ["A", "F", "C"])
                table_rows.append(row)

            df_segments = pd.DataFrame(table_rows)

            if not df_segments.empty:
                st.dataframe(df_segments, use_container_width=True)
            else:
                st.warning("Không có dữ liệu segment để hiển thị.")

            # ================= MODEL-LEVEL =================
            st.subheader("Model-level voting")

            model_votes = []

            for model_name, info in result["model_votes"].items():
                vote = info["model_vote"]
                model_votes.append(vote)
                st.markdown(f"- **{model_name}** → dự đoán **{vote}**")

            # ================= ENSEMBLE SUMMARY =================
            st.subheader("Ensemble voting summary")

            if model_votes:
                ensemble_count = (
                    pd.Series(model_votes)
                    .value_counts()
                    .reset_index()
                )
                ensemble_count.columns = ["Class", "Number of models"]

                st.dataframe(ensemble_count, use_container_width=True)

                final_votes = ensemble_count.iloc[0]["Number of models"]

                st.success(
                    f"Kết luận: lớp **{label}** được chọn với **{final_votes} / {len(model_votes)} model**"
                )
            else:
                st.warning("Không có model vote để ensemble.")

    st.markdown('</div>', unsafe_allow_html=True)


    


