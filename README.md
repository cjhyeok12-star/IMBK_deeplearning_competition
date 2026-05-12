# IMBK_deeplearning_competition
딥러닝(TabTransformer)을 활용한 은행 고객 이탈 예측 및 요인 분석 모델

## 프로젝트 명:

딥러닝 컴페티션

## 기간:

2026.05.12

## 기술 스택:

<img width="709" height="284" alt="image" src="https://github.com/user-attachments/assets/9d04e91d-0302-4836-ab97-94493c27b1d3" />


torch, torch.nn, TabTransformer, pandas, numpy, train_test_split, StandardScaler, LabelEncoder, OrdinalEncoder, compute_class_weight, accuracy_score, TensorDataset, DataLoader, copy, warnings을 사용했습니다.

## 데이터 전처리:

<img width="1023" height="574" alt="image" src="https://github.com/user-attachments/assets/810d454b-9b36-4d56-b41e-c98955e0050b" />


크게 **수입 대비 부담도** (부채 비율, 상환 부담 비율), **실질적인 현금 흐름** (가처분 소득, 잔고 비중, 투자 성향 비율), **창구 분석** (총 금융 창구 수, 창구당 평균 부채), **성숙도 확인** (나이 대비 신용 이력, 연체 심각도)를 활용하여 총 9개의 파생변수를 만들어 성능을 극대화 하였습니다.
신용평가는 상대적인 비율과 금융을 어떻게 활용하는지를 볼 수 있는 습관이 중요하다고 생각합니다. 그래서 딥러닝 모델이 스스로 학습하기 어려운 비율이나 곱셈을 도메인에 기반하여 정의하였습니다.

## EDA

<img width="465" height="690" alt="image" src="https://github.com/user-attachments/assets/76579fd7-070d-491f-8cc5-62b6c9f1795e" />

<img width="809" height="586" alt="image" src="https://github.com/user-attachments/assets/524b9b16-6b43-4702-82c2-ca064a67f251" />


EDA를 통해 데이터가 다중 분류이고 클래스 간에 불균형이 존재함을 파악하였습니다.
이를 해소하기 위해 학습 가중치를 부여하였고, 스케일링을 통해 모든 수치형 피처를 표준화하였습니다.
또한, Credit_Score가 3개의 등급으로 구성되어 있기에 출력층을 3개로 설정하였습니다.

## 모델링

<img width="649" height="286" alt="image" src="https://github.com/user-attachments/assets/9f9ee0dd-0ae1-4118-a72f-dc07d0a5b8f2" />


'ID', 'Customer_ID', 'Name', 'SSN'와 노이즈가 강하다고 보이는 비정형인 'Type_of_Loan'를 제거하여 성능을 높였고, 범주형은 5개, 연속형은 20여종의 변수(파생변수 포함)를 최종적으로 선택하였습니다.
모델은 TabTransformer를 채택하였고, 유니크 값을 자동으로 설계하였고, heads와 depth를 각각 8과 6으로 설정해 모델의 복잡도를 확보하였습니다, 또한 과적합 방지를 위해 dropout을 0.1로 적용하였습니다.

## 성능 결과

<img width="617" height="572" alt="image" src="https://github.com/user-attachments/assets/8c957c5d-af2e-4b02-90ea-d1a739adccb6" />


100번 반복 학습 결과, 현재 출력된 validation score는 0.78로 목표인 0.75이상을 충족하여 적절한 score를 뽑았다고 볼 수 있습니다.
