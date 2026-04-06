# [수정된 판독 로직]
conf_value = conf.item() * 100

if conf_value < 65.0:  # 신뢰도가 65% 미만인 경우
    st.error("⚠️ 판독 불충분 (Low Confidence)")
    st.warning("사진의 해상도가 낮거나 대상이 너무 멀리 있습니다.")
    st.markdown(f"""
        <div style='background-color: #fff3cd; padding: 1.5rem; border-radius: 15px; border-left: 5px solid #ffca28;'>
            <strong>검사 실패 사유:</strong> 신뢰도가 {conf_value:.1f}%로 너무 낮습니다.<br><br>
            <strong>해결 방법:</strong><br>
            1. 결함이 의심되는 부위를 <b>부분적으로 확대</b>하여 다시 찍어주세요.<br>
            2. 밝은 조명 아래에서 흔들림 없이 촬영해 주세요.<br>
            3. 본 AI는 부품 단위의 <b>근접 촬영</b>에 최적화되어 있습니다.
        </div>
        """, unsafe_allow_html=True)
    
if st.button("다시 업로드하기"):
    st.rerun() # 화면 초기화

else: # 신뢰도가 높을 때만 결과 표시
    if pred.item() == 1:
        res_label, res_color, res_icon = "⚠️ 결함 발견 (Defect)", "#E53935", "🚫"
    else:
        res_label, res_color, res_icon = "✅ 정상 (Normal)", "#43A047", "✔️"
    
    st.markdown(f'<p class="result-label" style="color: {res_color};">{res_icon} {res_label}</p>', unsafe_allow_html=True)
    st.metric(label="AI 신뢰도", value=f"{conf_value:.2f}%")
    st.progress(conf.item())
    
# ... (이하 생략) ...
