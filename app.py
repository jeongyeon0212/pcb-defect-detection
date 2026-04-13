import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import gdown

# 페이지 설정
st.set_page_config(page_title="PCB 결함 검출 AI", layout="wide")
st.title("🔍 PCB 결함 자동 검출 시스템")

# [세션 상태 초기화]
if 'retry_done' not in st.session_state:
    st.session_state.retry_done = False

# [전처리] 흑백 변환 + 중앙 확대 적용
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(400),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# [Grad-CAM 로직]
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
    for i in range(activations[0].shape[1]):
        activations[0][:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(activations[0], dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    h1.remove(); h2.remove()
    return heatmap

# [모델 로드]
@st.cache_resource
def load_model():
    model_path = 'pcb_model.pth'
    if not os.path.exists(model_path):
        file_id = '1RxWWMmFJwNonYVS-FSRzYeSML-pBb5FN'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    m.eval()
    return m

model = load_model()

# --- 메인 화면 구성 ---
uploaded_file = st.sidebar.file_uploader("PCB 이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    img_tensor.requires_grad = True
    
    # AI 판독
    output = model(img_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, pred = torch.max(probs, 1)
    conf_value = conf.item() * 100

    # ---------------------------------------------------------
    # 1. 결과 영역 (최상단 배치)
    # ---------------------------------------------------------
    st.markdown("### 📋 AI 판독 결과")
    
    # 신뢰도 낮을 때의 처리
    if conf_value < 70.0 and not st.session_state.retry_done:
        st.error(f"⚠️ 판독 불충분 (신뢰도: {conf_value:.1f}%)")
        st.warning("사진이 선명하지 않거나 학습 데이터와 다를 수 있습니다. 다시 촬영을 권장합니다.")
        if st.button("무시하고 결과 바로 보기"):
            st.session_state.retry_done = True
            st.rerun()
    else:
        # 최종 결과 표시 (신뢰도가 높거나 무시 버튼을 눌렀을 때)
        res_text = "⚠️ 결함 발견 (Defect)" if pred.item() == 1 else "✅ 정상 (Normal)"
        res_color = "#E53935" if pred.item() == 1 else "#43A047"
        
        # 큰 폰트로 결과 강조
        st.markdown(f"""
            <div style="background-color: {res_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h1 style="color: white; margin: 0;">{res_text}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # 신뢰도 메트릭
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("AI 신뢰도", f"{conf_value:.2f}%")
        with c2:
            st.write("") # 간격 맞춤
            st.progress(conf.item())

        st.session_state.retry_done = False # 결과 출력 후 상태 초기화

        st.write("---") # 구분선

        # ---------------------------------------------------------
        # 2. 이미지 및 시각화 영역 (하단 배치)
        # ---------------------------------------------------------
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ 원본 이미지")
            st.image(image, use_container_width=True, caption="업로드된 원본")

        with col2:
            st.subheader("🔥 판단 근거 시각화")
            # Grad-CAM 생성
            heatmap = get_gradcam(model, img_tensor, pred.item())
            img_np = np.array(image.resize((224, 224)))
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
            
            st.image(overlay, use_container_width=True, caption="AI가 집중한 영역 (빨간색)")

else:
    st.info("왼쪽 사이드바에서 이미지를 업로드하면 즉시 판독을 시작합니다.")
