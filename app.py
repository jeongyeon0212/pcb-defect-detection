import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import gdown

# --- [앱 설정 및 테마] ---
st.set_page_config(
    page_title="PCB 결함 검출 AI 포털",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [커스텀 CSS 스트일 - 앱 느낌 내기] ---
st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    .stApp {
        background-color: #f4f7f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* 제목 스타일 */
    .big-title {
        font-size: 2.8rem !important;
        font-weight: 800;
        color: #1A1D21;
        text-align: center;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: -1px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6D727A;
        text-align: center;
        margin-bottom: 3rem;
    }

    /* 카드 스타일 (결과창) */
    .result-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
        margin-bottom: 2rem;
    }
    
    /* 판독 결과 텍스트 */
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* 메트릭 스티일 커스텀 */
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #1E88E5 !important;
    }

    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e1e4e8;
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- [앱 상단 헤더] ---
st.markdown('<p class="big-title">🔍 PCB Defect AI Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">실시간 PCB 인공지능 결함 검출 시스템</p>', unsafe_allow_html=True)

# --- [사이드바 - 설정 및 업로드] ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2061/2061956.png", width=80) # PCB 아이콘
    st.header("🛠️ 검사 설정")
    st.markdown("AI 기반의 고정밀 PCB 결함 검출 시스템입니다.")
    uploaded_file = st.file_uploader("검사할 PCB 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
    st.info("💡 Tip: 전용 조명 아래에서 정면으로 촬영한 사진이 가장 정확합니다.")

# --- [핵심 로직: 모델 다운로드 및 Grad-CAM] ---
@st.cache_resource
def download_model():
    file_id = '1RSoxruMpDKfCsykVfrze2Ct4_nFF_Lt7'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'pcb_model.pth'
    if not os.path.exists(output):
        with st.spinner('🚀 최첨단 AI 모델을 클라우드에서 불러오는 중... (약 1분 소요)'):
            gdown.download(url, output, quiet=False)
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(output, map_location='cpu'))
    return m

def get_gradcam(model, img_tensor, label_idx):
    model.eval()
    target_layer = model.layer4[-1]
    gradients, activations = [], []
    def save_grad(grad): gradients.append(grad)
    def save_act(module, input, output): activations.append(output)
    h1 = target_layer.register_forward_hook(save_act)
    h2 = target_layer.register_full_backward_hook(lambda m, i, o: save_grad(o[0]))
    output = model(img_tensor)
    model.zero_grad()
    output[0, label_idx].backward()
    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations[0].shape[1]): activations[0][:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(activations[0], dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    h1.remove(); h2.remove()
    return heatmap

# --- [메인 화면 로직] ---
if not uploaded_file:
    # 파일을 업로드하기 전 보여줄 깔끔한 대기화면
    st.markdown("""
    <div style='text-align: center; padding: 5rem; background-color: white; border-radius: 20px; border: 2px dashed #e1e4e8; color: #6D727A;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3342/3342137.png' width='120' style='margin-bottom: 2rem;'/>
        <h3>PCB 이미지를 업로드하여 검사를 시작하세요</h3>
        <p>왼쪽 사이드바의 'Browse files' 버튼을 이용해 검사할 PCB 사진을 선택할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        model = download_model()
    except Exception as e:
        st.error(f"모델을 불러오는데 실패했습니다: {e}")
        st.stop()

    image = Image.open(uploaded_file).convert('RGB')
    
    # 레이아웃을 3컬럼으로 나누어 시각적 균형 맞춤
    col_img, col_res, col_cam = st.columns([1, 1.2, 1])
    
    with col_img:
        st.subheader("📸 원본 이미지")
        st.image(image, use_container_width=True, caption=uploaded_file.name)
        
    with col_res:
        st.subheader("📊 AI 판독 결과")
        st.markdown('<div class="result-card">', unsafe_allow_html=True) # 카드 시작
        
        # 전처리 및 판독
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        img_tensor.requires_grad = True
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
        # 결과에 따른 스타일 적용
        if pred.item() == 1:
            res_label, res_color, res_icon = "⚠️ 결함 발견 (Defect)", "#E53935", "🚫"
        else:
            res_label, res_color, res_icon = "✅ 정상 (Normal)", "#43A047", "✔️"
            
        st.markdown(f'<p class="result-label" style="color: {res_color};">{res_icon} {res_label}</p>', unsafe_allow_html=True)
        st.metric(label="AI 신뢰도 (Confidence)", value=f"{conf.item()*100:.2f}%")
        st.progress(conf.item()) # 시각적인 바 추가
        st.markdown('</div>', unsafe_allow_html=True) # 카드 끝
        
    with col_cam:
        st.subheader("🗺️ 판단 근거 (Grad-CAM)")
        heatmap = get_gradcam(model, img_tensor, pred.item())
        img_np = np.array(image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        st.image(overlay, use_container_width=True, caption="빨간 영역이 AI가 집중한 곳입니다.")

    # 하단 바
    st.markdown("---")
    st.caption("© 2024 PCB Defect Detection Project | SKKU | Powered by Streamlit")
