import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 페이지 설정
st.set_page_config(page_title="PCB 결함 검출 AI", layout="wide")
st.title("🔍 PCB 결함 자동 검출 시스템 (+Grad-CAM)")

# [Grad-CAM 로직] AI가 주목한 위치를 계산합니다.
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

# 모델 로드 (가중치 파일 연결)
@st.cache_resource
def load_model():
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load('pcb_model.pth', map_location='cpu'))
    return m

model = load_model()
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("AI 판독 결과")
        res = "⚠️ 결함 발견 (Defect)" if pred.item() == 1 else "✅ 정상 (Normal)"
        color = "red" if pred.item() == 1 else "green"
        st.markdown(f"<h2 style='color: {color};'>{res}</h2>", unsafe_allow_html=True)
        st.metric("신뢰도 (Confidence)", f"{conf.item()*100:.2f}%")
        
        # Grad-CAM 결과 출력
        heatmap = get_gradcam(model, img_tensor, pred.item())
        img_np = np.array(image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        st.subheader("판단 근거 시각화 (Grad-CAM)")
        st.image(overlay, caption="빨간 부분이 AI가 집중한 곳입니다.", use_container_width=True)
