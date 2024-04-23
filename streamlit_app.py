import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
from streamlit.components.v1 import html

# 스트림릿 앱 제목 설정
st.title('사용자 분석 및 예측')

# 데이터 URL
data_url = 'https://raw.githubusercontent.com/dongcoding7/std-b-2/main/users.6M0xxK.2020.public_%EB%82%98%EB%9D%BC%EC%88%98%EC%A0%95_ver3.csv'

# 데이터 불러오기 및 캐싱
data = pd.read_csv(data_url)

# 파일 업로더
uploaded_file = st.file_uploader("CSV 파일 선택", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# 데이터프레임 표시
st.dataframe(data.head())

# 사용자 역할 결정 함수 정의
def determine_role(row):
    if row['productsSold'] > row['productsBought']:
        return 'seller'
    elif row['productsSold'] < row['productsBought']:
        return 'buyer'
    else:
        return 'both'

# 새로운 역할 컬럼 생성 및 데이터 전처리
data['role'] = data.apply(determine_role, axis=1)
data = data[data['role'] != 'both']
data['is_seller'] = (data['role'] == 'seller').astype(int)
data['is_buyer'] = (data['role'] == 'buyer').astype(int)

# 필요한 특성 선택
features = [
    'socialNbFollowers', 'socialNbFollows', 'socialProductsLiked', 'productsListed',
    'productsPassRate', 'productsWished'
]
X = data[features]
y = data[['is_seller', 'is_buyer']]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 모델 생성 및 훈련
model = LogisticRegression()
model.fit(X_train_scaled, y_train['is_seller'])
model_buyer = LogisticRegression()
model_buyer.fit(X_train_scaled, y_train['is_buyer'])

# SHAP Explainer 인스턴스 생성
explainer_seller = shap.LinearExplainer(model, X_train_scaled)
explainer_buyer = shap.LinearExplainer(model_buyer, X_train_scaled)

# 새로운 데이터에 대한 예측 함수
def predict_new_data(new_data_scaled, model_seller, model_buyer):
    proba_seller = model_seller.predict_proba(new_data_scaled)[0, 1]
    proba_buyer = model_buyer.predict_proba(new_data_scaled)[0, 1]
    return proba_seller, proba_buyer

# 예측 결과 표시
def display_prediction(proba_seller, proba_buyer):
    st.write(f"이 고객이 판매자일 확률: {proba_seller*100:.2f}%")
    st.write(f"이 고객이 구매자일 확률: {proba_buyer*100:.2f}%")

# 예측 실행 버튼
if st.sidebar.button('예측 실행'):
    # 사용자 입력 데이터 가져오기
    new_data = {
        'socialNbFollowers': st.sidebar.number_input('소셜 팔로워 수', value=10),
        'socialNbFollows': st.sidebar.number_input('소셜 팔로우 수', value=5),
        'socialProductsLiked': st.sidebar.number_input('좋아한 상품 수', value=3),
        'productsListed': st.sidebar.number_input('상품 목록 수', value=4),
        'productsPassRate': st.sidebar.number_input('상품 통과율', value=80.0),
        'productsWished': st.sidebar.number_input('상품 희망 수', value=2)
    }
    # 입력 데이터 스케일링
    new_data_scaled = scaler.transform([list(new_data.values())])
    
    # 예측 결과 계산
    proba_seller, proba_buyer = predict_new_data(new_data_scaled, model, model_buyer)
    
    # 예측 결과 표시
    display_prediction(proba_seller, proba_buyer)
    
    # SHAP 값을 계산하여 시각화 생성
    shap_values_seller = explainer_seller.shap_values(new_data_scaled)
    shap_values_buyer = explainer_buyer.shap_values(new_data_scaled)
    
    # 판매자와 구매자 모델의 SHAP force plot을 생성하고 HTML로 변환
    force_plot_html_seller = shap.force_plot(explainer_seller.expected_value, shap_values_seller[0], features, show=False, matplotlib=False)
    force_plot_html_buyer = shap.force_plot(explainer_buyer.expected_value, shap_values_buyer[0], features, show=False, matplotlib=False)

    # SHAP force plot을 HTML로 변환
    shap_html_seller = f"<head>{shap.getjs()}</head><body>{force_plot_html_seller.html()}</body>"
    shap_html_buyer = f"<head>{shap.getjs()}</head><body>{force_plot_html_buyer.html()}</body>"

    # Streamlit에 SHAP force plot 내장
    st.write("판매자 SHAP Force Plot:")
    html(shap_html_seller, height=300)
    st.write("구매자 SHAP Force Plot:")
    html(shap_html_buyer, height=300)
