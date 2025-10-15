import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Hệ thống Dự đoán Bệnh",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Gemini Analysis Function ---
def analyze_with_gemini(api_key, disease_type, probability, input_data, shap_summary):
    """Phân tích kết quả dự đoán bằng Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-latest')
        
        # Format the input data and SHAP summary for the prompt
        data_str = ", ".join([f"{k}: {v}" for k, v in input_data.to_dict('records')[0].items()])
        
        prompt = f"""
        Bạn là một trợ lý y tế AI. Nhiệm vụ của bạn là giải thích kết quả dự đoán bệnh cho một người dùng không có chuyên môn y tế.
        
        Bối cảnh:
        - Loại bệnh được dự đoán: {disease_type.replace('_', ' ').title()}
        - Dữ liệu sức khỏe của người dùng được đưa vào: {data_str}
        - Xác suất mắc bệnh do mô hình AI dự đoán: {probability:.2%}
        - Các yếu tố ảnh hưởng lớn nhất đến dự đoán (từ phân tích SHAP): {shap_summary}

        Yêu cầu:
        1. Đưa ra một lời giải thích rõ ràng và dễ hiểu về dữ liệu sức khoẻ của người dùng được đưa vào và ý nghĩa của xác suất dự đoán.
        2. Dựa vào các yếu-tố-ảnh-hưởng-lớn-nhất và dữ-liệu-sức-khỏe, giải thích tại sao mô hình lại đưa ra dự đoán như vậy. Hãy nói về việc các chỉ số của người dùng (cao hay thấp) đã góp phần vào kết quả này như thế nào.
        3. Cung cấp một số lời khuyên chung về lối sống hoặc các bước tiếp theo mà người dùng nên cân nhắc. Nhấn mạnh rằng đây không phải là chẩn đoán y tế và người dùng nên tham khảo ý kiến bác sĩ.
        4. Trình bày câu trả lời bằng tiếng Việt, với giọng văn thân thiện, quan tâm và chuyên nghiệp.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Không thể kết nối đến Gemini. Vui lòng kiểm tra lại API Key và kết nối mạng. Lỗi: {e}"

# --- Functions ---
def load_model(disease_type):
    """Tải mô hình đã được huấn luyện."""
    model_path = os.path.join('model', f'{disease_type}_model.joblib')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    return None

def get_feature_names(disease_type):
    """Lấy tên các đặc trưng cho từng loại bệnh."""
    if disease_type == 'heart':
        return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    elif disease_type == 'diabetes':
        return ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    elif disease_type == 'breast_cancer':
        # Using a subset of important features for simplicity in the UI
        return [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
    return []

def create_input_form(disease_type):
    """Tạo form nhập liệu động."""
    feature_names = get_feature_names(disease_type)
    inputs = {}
    with st.form(key=f"{disease_type}_form"):
        st.subheader(f"Nhập thông tin cho dự đoán bệnh {disease_type.replace('_', ' ').title()}")
        
        # Create columns for better layout
        if len(feature_names) > 10:
            cols = st.columns(3)
        elif len(feature_names) > 1:
            cols = st.columns(2)
        else:
            cols = [st]

        for i, feature in enumerate(feature_names):
            with cols[i % len(cols)]:
                inputs[feature] = st.number_input(
                    label=feature.replace('_', ' ').title(), 
                    key=f"{disease_type}_{feature}",
                    step=0.1,
                    format="%.1f"
                )
        
        submit_button = st.form_submit_button(label="Dự đoán")
    return inputs, submit_button

def get_advice(disease_type, probability):
    """Đưa ra lời khuyên dựa trên kết quả."""
    advice = ""
    if probability > 0.7:
        advice = f"**Nguy cơ cao!** Xác suất mắc bệnh {disease_type.replace('_', ' ')} là rất cao. Bạn nên tham khảo ý kiến bác sĩ ngay lập tức để được chẩn đoán và tư vấn chi tiết."
    elif probability > 0.4:
        advice = f"**Cảnh báo!** Bạn có một số dấu hiệu nguy cơ. Mặc dù chưa chắc chắn, bạn nên duy trì lối sống lành mạnh và theo dõi sức khỏe định kỳ. Tham khảo ý kiến bác sĩ nếu có bất kỳ triệu chứng nào."
    else:
        advice = f"**Nguy cơ thấp.** Dựa trên các chỉ số hiện tại, nguy cơ mắc bệnh {disease_type.replace('_', ' ')} của bạn là thấp. Hãy tiếp tục duy trì lối sống lành mạnh."
    return advice

def show_prediction_results(input_df, model, disease_type, api_key):
    """Hiển thị kết quả dự đoán, SHAP và lời khuyên."""
    # 1. Predict
    prediction_proba = model.model.predict_proba(input_df)[0]
    probability_of_disease = prediction_proba[1]  # Probability of class 1 (disease)

    # 2. Display Results
    st.header("Kết quả Dự đoán")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label=f"Xác suất mắc bệnh {disease_type.replace('_', ' ').title()}",
            value=f"{probability_of_disease:.2%}"
        )
        st.progress(probability_of_disease)
        
        advice = get_advice(disease_type, probability_of_disease)
        st.info(advice)

    with col2:
        # 3. Explain with SHAP
        st.subheader("Giải thích Kết quả với SHAP")
        shap_summary_for_gemini = "Không có"
        try:
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer(input_df)
            
            # Extract feature names and their SHAP values for Gemini prompt
            shap_dict = dict(zip(input_df.columns, shap_values.values[0][:, 1]))
            # Sort by absolute value to find most influential features
            sorted_shap = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
            shap_summary_for_gemini = ", ".join([f"{item[0]} ({'tăng nguy cơ' if item[1] > 0 else 'giảm nguy cơ'})" for item in sorted_shap[:5]])

            shap.plots.force(
                shap_values.base_values[0][1],
                shap_values.values[0][:, 1],
                input_df,
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf(), bbox_inches='tight')
            plt.close()

            st.markdown("---")
            st.write("**Giải thích biểu đồ:** Biểu đồ SHAP ở trên cho thấy các yếu tố đẩy dự đoán về phía 'Mắc bệnh' (màu đỏ) và các yếu tố đẩy về phía 'Không mắc bệnh' (màu xanh).")

        except Exception as e:
            st.error(f"Không thể tạo giải thích SHAP: {e}")

    # 4. Gemini Analysis
    if api_key:
        with st.spinner("AI đang phân tích kết quả, vui lòng chờ..."):
            gemini_analysis = analyze_with_gemini(
                api_key,
                disease_type,
                probability_of_disease,
                input_df,
                shap_summary_for_gemini
            )
            with st.expander("🔍 Xem Phân tích Chi tiết từ AI (Gemini)", expanded=True):
                st.markdown(gemini_analysis)
    else:
        st.warning("Nhập API Key Gemini của bạn ở thanh bên để nhận phân tích chi tiết.")


# --- Main Application ---
st.title("🩺 Hệ thống Hỗ trợ Dự đoán và Giải thích Bệnh")

# Sidebar for navigation
with st.sidebar:
    st.header("Cấu hình")
    gemini_api_key = st.text_input("Nhập API Key Gemini của bạn", type="password", help="Lấy API Key của bạn từ Google AI Studio.")

    # Chức năng chẩn đoán model
    if gemini_api_key:
        st.subheader("Kiểm tra Model tương thích")
        if st.button("Liệt kê các Model có sẵn"):
            try:
                genai.configure(api_key=gemini_api_key)
                models_list = []
                with st.spinner("Đang lấy danh sách model..."):
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            models_list.append(m.name)
                st.success("Các model có thể sử dụng:")
                st.write(models_list)
            except Exception as e:
                st.error(f"Lỗi khi liệt kê model: {e}")
    
    st.header("Chọn chức năng")
    disease_choice = st.selectbox(
        "Chọn loại bệnh để dự đoán:",
        ('heart', 'diabetes', 'breast_cancer'),
        format_func=lambda x: x.replace('_', ' ').title()
    )

# Main content
model = load_model(disease_choice)
feature_names = get_feature_names(disease_choice)

if not model:
    st.error(f"Không tìm thấy mô hình cho bệnh {disease_choice.replace('_', ' ').title()}. Vui lòng huấn luyện mô hình trước.")
else:
    # Create two tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Nhập thủ công", "📋 Dán dữ liệu"])

    with tab1:
        # Manual form input
        user_inputs, submitted_form = create_input_form(disease_choice)
        if submitted_form:
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df[feature_names]  # Ensure column order
            show_prediction_results(input_df, model, disease_choice, gemini_api_key)

    with tab2:
        # Paste data input
        st.subheader("Dán dữ liệu thô (phân tách bằng dấu phẩy)")
        
        # Provide placeholder based on selected disease
        placeholder_examples = {
            'heart': 'Ví dụ: 63,1,3,145,233,1,0,150,0,2.3,0,0,1',
            'diabetes': 'Ví dụ: 6,148,72,35,0,33.6,0.627,50',
            'breast_cancer': 'Ví dụ: 17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189'
        }
        pasted_data = st.text_area(
            "Dữ liệu:", 
            placeholder=placeholder_examples.get(disease_choice, ""), 
            height=100,
            key=f"paste_area_{disease_choice}"
        )
        submitted_paste = st.button("Dự đoán từ dữ liệu đã dán", key=f"paste_submit_{disease_choice}")

        if submitted_paste:
            if pasted_data:
                try:
                    # Parse the pasted data
                    data_values = [float(x.strip()) for x in pasted_data.split(',')]
                    
                    # Validate number of features
                    if len(data_values) == len(feature_names):
                        input_dict = dict(zip(feature_names, data_values))
                        input_df = pd.DataFrame([input_dict])
                        show_prediction_results(input_df, model, disease_choice, gemini_api_key)
                    else:
                        st.error(f"Lỗi: Số lượng giá trị không khớp. Mô hình cho bệnh '{disease_choice.replace('_', ' ').title()}' yêu cầu {len(feature_names)} giá trị, nhưng bạn đã cung cấp {len(data_values)}.")
                except ValueError:
                    st.error("Lỗi: Vui lòng đảm bảo tất cả các giá trị đều là số và được phân tách bằng dấu phẩy.")
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi không mong muốn: {e}")
            else:
                st.warning("Vui lòng dán dữ liệu vào ô bên trên.")

# Add a footer
st.markdown("---")
st.write("Lưu ý: Kết quả chỉ mang tính tham khảo và không thay thế cho chẩn đoán y tế chuyên nghiệp.")
