import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import gdown  # 1. import는 맨 위로 올렸습니다.

# 페이지 설정
st.set_page_config(page_title="PCB 결함 검출 AI", layout="wide")
st.title("🔍 PCB 결함 자동 검출 시스템 (+Grad-CAM)")

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

# [모델 로드 로직] 구글 드라이브 다운로드 포함
@st.cache_resource
def load_model():
    model_path = 'pcb_model.pth'
    
    # 파일이 없으면 구글 드라이브에서 다운로드
    if not os.path.exists(model_path):
        with st.spinner('구글 드라이브에서 모델 가중치를 다운로드 중입니다... (최초 1회)'):
            file_id = '1RxWWMmFJwNonYVS-FSRzYeSML-pBb5FN'
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"다운로드 중 오류가 발생했습니다: {e}")
                return None

    # 모델 구조 생성 및 가중치 입히기
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    m.eval()
    return m

model = load_model()
if 'retry_done' not in st.session_state:
    st.session_state.retry_done = False

# 모델 로드가 완료된 경우에만 UI 출력
if model:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # 컬러를 흑백으로 (중요!)
        transforms.Resize(400),                      # 전체적으로 키운 뒤
        transforms.CenterCrop(224),                  # 중앙부만 줌인(Zoom-in)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    uploaded_file = st.sidebar.file_uploader("PCB 이미지 업로드", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        img_tensor.requires_grad = True
        
        # AI 판독 시작
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        conf_value = conf.item() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("AI 판독 결과")
            
            # [수정된 판독 로직]
        if conf_value < 70.0 and not st.session_state.retry_done:
            # 첫 번째 저신뢰도 발생 시: 경고만 띄우고 결과 안 보여줌
            st.error("⚠️ 판독 불충분 (Low Confidence)")
            st.warning(f"현재 신뢰도가 {conf_value:.1f}%로 낮습니다. 사진을 더 가까이, 선명하게 찍어 다시 업로드해 주세요.")
            
            if st.button("무시하고 결과 바로 보기"):
                st.session_state.retry_done = True
                st.rerun() # 다시 실행해서 아래 else문으로 진입하게 함
        
        else:
            # 신뢰도가 높거나, 사용자가 '무시하고 보기'를 눌렀을 때만 결과 출력
            res = "⚠️ 결함 발견 (Defect)" if pred.item() == 1 else "✅ 정상 (Normal)"
            color = "#E53935" if pred.item() == 1 else "#43A047"
            
            st.markdown(f"<h2 style='color: {color};'>{res}</h2>", unsafe_allow_html=True)
            st.metric("신뢰도 (Confidence)", f"{conf_value:.2f}%")
            st.progress(conf.item())
            
            # 결과 출력 후에는 다시 초기화 (다음 사진을 위해)
            st.session_state.retry_done = False


                # Grad-CAM 시각화
                heatmap = get_gradcam(model, img_tensor, pred.item())
                img_np = np.array(image.resize((224, 224)))
                heatmap_resized = cv2.resize(heatmap, (224, 224))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
                
                st.write("---")
                st.subheader("판단 근거 시각화 (Grad-CAM)")
                st.image(overlay, caption="빨간색/노란색 영역이 AI가 집중한 부분입니다.", use_container_width=True)
    else:
        st.info("왼쪽 사이드바에서 이미지를 업로드해 주세요.")
