import numpy as np
import pandas as pd
import pandas_datareader as pdr
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt

Stock_Ticker = 'VSS'                                                # Stock Ticker :  Readme 보삼, 'DBC' 대신 다른 티커 넣으면 됨                                               

# 크롤링하려는 범위, Start time이 과거여도 가용되는 데이터부터 긁어옴            
start = dt.datetime(1960, 1, 1)
end = dt.datetime.today()

# yahoo finance에서 데이터를 긁어옴 interval='m'이면 월변화를 가져옴
RAW_DATA = pdr.get_data_yahoo([Stock_Ticker], start, end, interval='m')    # start부터 end까지 판다스 data frame 형태로 저장됨
DP = RAW_DATA.Close.copy()                                                 # 해당 데이터에서 Close(종가) 열만 DP라는 data frame으로 카피함
print(DP.tail())                                                           # 제일 최근 5개를 보여줌. DP.head() 함수는 젤위에 5개를 보여줌

# 무시하삼...
def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

Ans = yes_or_no("마지막 꺼 잘봐 지울겨?")                           # 이건 회사에서 알려줄게
if Ans :
    DP.drop(DP.index[-1], inplace=True)
print("마지막꺼 지웠다.")
print(DP.tail())
Ans = yes_or_no("기회 또 준다 다시봐봐 지울겨?")
if Ans :
    DP.drop(DP.index[-1], inplace=True)
print("마지막꺼 지웠다.")
print(DP.tail())

# 이건 1개월 차이부터 12개월 차이까지 계산해서 다른 열로 만듦
# Calculate Continuously Complouned Return
DP['1M'] = np.nan
for i in range(1, len(DP)):
    # DP.iloc[i,1] = (DP.iloc[i,0] - DP.iloc[i-1, 0])/DP.iloc[i-1, 0]
    DP.iloc[i,1] = np.log(DP.iloc[i,0] / DP.iloc[i-1, 0])
DP['2M'] = np.nan
for i in range(2, len(DP)):
    DP.iloc[i,2] = np.log(DP.iloc[i,0] / DP.iloc[i-2, 0])
DP['3M'] = np.nan
for i in range(3, len(DP)):
    DP.iloc[i,3] = np.log(DP.iloc[i,0] / DP.iloc[i-3, 0])
DP['4M'] = np.nan
for i in range(4, len(DP)):
    DP.iloc[i,4] = np.log(DP.iloc[i,0] / DP.iloc[i-4, 0])
DP['5M'] = np.nan
for i in range(5, len(DP)):
    DP.iloc[i,5] = np.log(DP.iloc[i,0] / DP.iloc[i-5, 0])
DP['6M'] = np.nan
for i in range(6, len(DP)):
    DP.iloc[i,6] = np.log(DP.iloc[i,0] / DP.iloc[i-6, 0])
DP['7M'] = np.nan
for i in range(7, len(DP)):
    DP.iloc[i,7] = np.log(DP.iloc[i,0] / DP.iloc[i-7, 0])
DP['8M'] = np.nan
for i in range(8, len(DP)):
    DP.iloc[i,8] = np.log(DP.iloc[i,0] / DP.iloc[i-8, 0])
DP['9M'] = np.nan
for i in range(9, len(DP)):
    DP.iloc[i,9] = np.log(DP.iloc[i,0] / DP.iloc[i-9, 0])
DP['10M'] = np.nan
for i in range(10, len(DP)):
    DP.iloc[i,10] = np.log(DP.iloc[i,0] / DP.iloc[i-10, 0])
DP['11M'] = np.nan
for i in range(11, len(DP)):
    DP.iloc[i,11] = np.log(DP.iloc[i,0] / DP.iloc[i-11, 0])
DP['12M'] = np.nan
for i in range(12, len(DP)):
    DP.iloc[i,12] = np.log(DP.iloc[i,0] / DP.iloc[i-12, 0])
DP['Future'] = np.nan
for i in range(0, len(DP)-1):
    DP.iloc[i,13] = np.log(DP.iloc[i+1,0] / DP.iloc[i, 0])

print(DP.tail())

# Least Squre를 쓰고자 하는 기간 설정
interval = 10             # Years
DP_INTERVAL = DP.iloc[-12*interval-1:,:].copy()            # DP 데이터프레임에서 10년치를 복사해서 새로 저장

# Normalization value/STD for each columns
DP_INTERVAL['1M'] = DP_INTERVAL['1M'] / 1.0
DP_INTERVAL['2M'] = DP_INTERVAL['2M'] / 2.0
DP_INTERVAL['3M'] = DP_INTERVAL['3M'] / 3.0
DP_INTERVAL['4M'] = DP_INTERVAL['4M'] / 4.0
DP_INTERVAL['5M'] = DP_INTERVAL['5M'] / 5.0
DP_INTERVAL['6M'] = DP_INTERVAL['6M'] / 6.0
DP_INTERVAL['7M'] = DP_INTERVAL['7M'] / 7.0
DP_INTERVAL['8M'] = DP_INTERVAL['8M'] / 8.0
DP_INTERVAL['9M'] = DP_INTERVAL['9M'] / 9.0
DP_INTERVAL['10M'] = DP_INTERVAL['10M'] / 10.0
DP_INTERVAL['11M'] = DP_INTERVAL['11M'] / 11.0
DP_INTERVAL['12M'] = DP_INTERVAL['12M'] / 12.0

# Normalization based on the standard deviation from the data
print(DP_INTERVAL['1M'].std())
print(DP_INTERVAL['2M'].std())
print(DP_INTERVAL['3M'].std())
print(DP_INTERVAL['4M'].std())
print(DP_INTERVAL['5M'].std())
print(DP_INTERVAL['6M'].std())
print(DP_INTERVAL['7M'].std())
print(DP_INTERVAL['8M'].std())
print(DP_INTERVAL['9M'].std())
print(DP_INTERVAL['10M'].std())
print(DP_INTERVAL['11M'].std())
print(DP_INTERVAL['12M'].std())
DP_INTERVAL['1M'] = DP_INTERVAL['1M'] / DP_INTERVAL['1M'].std()
DP_INTERVAL['2M'] = DP_INTERVAL['2M'] / DP_INTERVAL['2M'].std()
DP_INTERVAL['3M'] = DP_INTERVAL['3M'] / DP_INTERVAL['3M'].std()
DP_INTERVAL['4M'] = DP_INTERVAL['4M'] / DP_INTERVAL['4M'].std()
DP_INTERVAL['5M'] = DP_INTERVAL['5M'] / DP_INTERVAL['5M'].std()
DP_INTERVAL['6M'] = DP_INTERVAL['6M'] / DP_INTERVAL['6M'].std()
DP_INTERVAL['7M'] = DP_INTERVAL['7M'] / DP_INTERVAL['7M'].std()
DP_INTERVAL['8M'] = DP_INTERVAL['8M'] / DP_INTERVAL['8M'].std()
DP_INTERVAL['9M'] = DP_INTERVAL['9M'] / DP_INTERVAL['9M'].std()
DP_INTERVAL['10M'] = DP_INTERVAL['10M'] / DP_INTERVAL['10M'].std()
DP_INTERVAL['11M'] = DP_INTERVAL['11M'] / DP_INTERVAL['11M'].std()
DP_INTERVAL['12M'] = DP_INTERVAL['12M'] / DP_INTERVAL['12M'].std()


# 판다스 Data Frame을 numpy array로 변환
DP_NP = DP_INTERVAL.to_numpy()
x_train_np = DP_NP[:-1,1:13]    # Training Set 설정 (1M ~ 12M열)
y_train_np = DP_NP[:-1,-1]      # Training Set 설정 (Future열)

x_test_np = DP_NP[-1,1:13]      # Estimation 

x_train = torch.from_numpy(x_train_np).float()     # numpy -> pytorch로 간다
y_train = torch.from_numpy(y_train_np).float()     # numpy -> pytorch로 간다
y_train = y_train.view([-1,1])                     # 행벡터 -> 열벡터로


# 리스트 스퀘어쓰는 클래스 정의하는 거임.
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 1)             # 만약에 개월수 변경하고 싶으면 수치 변경하면 됨. 근데 train set도 수정해야함.

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)   # lr 원하는 수치로 변경, 경사하강법임

nb_epochs = 700000                                           # iteration 개수 ... 많이 필요하드라...
for epoch in range(nb_epochs+1):
    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 50000 == 0:
    # 50000번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
print('헉헉헉')

[coeff, bias] = model.parameters()                            # 
Err_COV = cost.item()

x_test = torch.from_numpy(x_test_np).float()
x_test = x_test.view([-1,1])
# y_est = (coeff @ x_test + bias) / np.sqrt(Err_COV)
y_est = (coeff @ x_test + bias)
y_est_np = y_est.cpu().data.numpy()
print('='*50)
print('Stock Ticker:{}  /  Result:{}'.format(Stock_Ticker, y_est_np[0][0]))
print('Standard Deviation :{}'.format(np.sqrt(Err_COV)))
print('='*50)