import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import streamlit.components.v1 as components

# Streamlit 앱의 타이틀 설정
st.title('사용자 분석 및 예측')

# 데이터 로드 함수
@st.experimental_memo  # 이 부분을 `@st.cache_data`로 변경
def load_data(url):
    return pd.read_csv(url)

# GitHub Raw URL 고정
data_url = 'https://raw.githubusercontent.com/dongcoding7/std-b-2/main/users.6M0xxK.2020.public_%EB%82%98%EB%9D%BC%EC%88%98%EC%A0%95_ver3.csv'
data = load_data(data_url)

# 파일 업로더
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# 데이터프레임 표시
st.dataframe(data.head(), height=95)

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

# 결과 표시를 위한 자리 확보
results_placeholder = st.empty()

# 새로운 데이터에 대한 예측 함수
def predict_new_data(new_data_scaled, model_seller, model_buyer):
    proba_seller = model_seller.predict_proba(new_data_scaled)[0, 1]
    proba_buyer = model_buyer.predict_proba(new_data_scaled)[0, 1]
    return proba_seller, proba_buyer

# 새로운 데이터에 대한 예측 및 시각화를 실행하는 함수
def predict_and_visualize():
    # 입력 데이터를 DataFrame으로 만듦
    new_data = pd.DataFrame({
        'socialNbFollowers': [socialNbFollowers],
        'socialNbFollows': [socialNbFollows],
        'socialProductsLiked': [socialProductsLiked],
        'productsListed': [productsListed],
        'productsPassRate': [productsPassRate],
        'productsWished': [productsWished]
    })

    # 새 데이터 스케일링
    new_data_scaled = scaler.transform(new_data)
    
    # 예측 실행
    proba_seller, proba_buyer = predict_new_data(new_data_scaled, model, model_buyer)

    # 결과 자리에 시각화 업데이트
    with results_placeholder.container():
        st.write(f"이 고객이 판매자일 확률: {proba_seller*100:.2f}%")
        st.write(f"이 고객이 구매자일 확률: {proba_buyer*100:.2f}%")
        # SHAP 시각화 코드를 여기에 넣으세요

# 사이드바 제목
st.sidebar.header('새 고객 데이터 입력')

# 사이드바에 입력 필드 배치
socialNbFollowers = st.sidebar.number_input('소셜 팔로워 수', min_value=0, value=8)
socialNbFollows = st.sidebar.number_input('소셜 팔로우 수', min_value=0, value=3)
socialProductsLiked = st.sidebar.number_input('좋아한 상품 수', min_value=0, value=1)
productsListed = st.sidebar.number_input('상품 목록 수', min_value=0, value=1)
productsPassRate = st.sidebar.number_input('상품 통과율', min_value=0, value=50)
productsWished = st.sidebar.number_input('상품 희망 수', min_value=0, value=5)

# 예측 및 시각화 실행 버튼
if st.sidebar.button('예측 및 시각화 실행'):
    predict_and_visualize()

# 노트북에 JS 시각화 코드 로드
shap.initjs()

# SHAP 시각화를 위한 준비
# 모델이 생성되었고, 데이터가 준비된 후에 SHAP 설명자를 생성합니다.
explainer_seller = shap.LinearExplainer(model, X_train_scaled, feature_dependence="independent")
explainer_buyer = shap.LinearExplainer(model_buyer, X_train_scaled, feature_dependence="independent")

# 앱이 처음 실행될 때 예측 및 시각화 수행
if 'initialized' not in st.session_state:
    predict_and_visualize()
    st.session_state['initialized'] = True
