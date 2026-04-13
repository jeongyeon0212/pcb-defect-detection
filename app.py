import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

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

# 모델 로드 (파일 유무 체크 추가)
@st.cache_resource
import gdown

# 모델 로드 (구글 드라이브에서 다운로드 로직 추가)
@st.cache_resource
def load_model():
    model_path = 'pcb_model.pth'
    
    # 파일이 없을 경우 구글 드라이브에서 다운로드
    if not os.path.exists(model_path):
        with st.spinner('구글 드라이브에서 모델 파일을 가져오고 있습니다. 잠시만 기다려 주세요...'):
            # 구글 드라이브 공유 링크의 ID 부분만 사용
            file_id = '1RxWWMmFJwNonYVS-FSRzYeSML-pBb5FN'
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"모델 다운로드 중 오류 발생: {e}")
                return None

    # 모델 구조 정의 및 가중치 로드
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    m.eval() # 평가 모드 설정
    return m
    
    m = models.resnet18(weights=None) # 최신 버전 표기법
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    return m

model = load_model()

# 모델 로드 성공 시에만 진행
if model:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("AI 판독 결과")
            
            # 신뢰도에 따른 분기 처리 (이 부분이 if uploaded_file 안에 있어야 합니다)
            if conf_value < 65.0:
                st.error("⚠️ 판독 불충분 (Low Confidence)")
                st.warning("사진의 해상도가 낮거나 대상이 너무 멀리 있습니다.")
                st.markdown(f"""
                    <div style='background-color: #fff3cd; padding: 1.5rem; border-radius: 15px; border-left: 5px solid #ffca28; color: black;'>
                        <strong>검사 실패 사유:</strong> 신뢰도가 {conf_value:.1f}%로 너무 낮습니다.<br><br>
                        <strong>해결 방법:</strong><br>
                        1. 결함 부위를 <b>확대 촬영</b>해 주세요.<br>
                        2. 밝은 조명을 활용해 주세요.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                res = "⚠️ 결함 발견 (Defect)" if pred.item() == 1 else "✅ 정상 (Normal)"
                color = "#E53935" if pred.item() == 1 else "#43A047"
                st.markdown(f"<h2 style='color: {color};'>{res}</h2>", unsafe_allow_html=True)
                st.metric("신뢰도 (Confidence)", f"{conf_value:.2f}%")
                st.progress(conf.item())

                # Grad-CAM 결과 출력
                heatmap = get_gradcam(model, img_tensor, pred.item())
                img_np = np.array(image.resize((224, 224)))
                heatmap_resized = cv2.resize(heatmap, (224, 224))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
                
                st.write("---")
                st.subheader("판단 근거 시각화 (Grad-CAM)")
