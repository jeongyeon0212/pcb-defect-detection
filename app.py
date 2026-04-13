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


    # 1. 결과 영역 (슬림 버전)
    # ---------------------------------------------------------
    st.markdown("### 📋 AI 판독 리포트")
    
    if conf_value < 70.0 and not st.session_state.retry_done:
        st.error(f"⚠️ 판독 불충분 (신뢰도: {conf_value:.1f}%)")
        st.info("사진을 더 가깝고 선명하게 찍어주시면 정확도가 올라갑니다.")
        if st.button("무시하고 결과 바로 보기"):
            st.session_state.retry_done = True
            st.rerun()
    else:
        res_text = "결함 발견 (Defect)" if pred.item() == 1 else "정상 (Normal)"
        res_color = "#E53935" if pred.item() == 1 else "#43A047"
        res_icon = "⚠️" if pred.item() == 1 else "✅"
        
        # 가로로 배치: 결과 텍스트와 신뢰도를 한 줄에 표시
        col_res, col_met = st.columns([2, 1])
        
        with col_res:
            # 박스 크기를 줄이고 테두리 위주로 디자인
            st.markdown(f"""
                <div style="border: 2px solid {res_color}; padding: 5px; border-radius: 10px; text-align: center;">
                    <h2 style="color: {res_color}; margin: 0;">{res_icon} {res_text}</h2>
                </div>
            """, unsafe_allow_html=True)
            
        with col_met:
            st.metric("AI 신뢰도", f"{conf_value:.2f}%")

        st.progress(conf.item())
        st.session_state.retry_done = False 

        st.write("") # 약간의 간격
        st.write("---")
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
