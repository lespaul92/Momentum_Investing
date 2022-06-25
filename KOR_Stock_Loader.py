import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime as dt

# 크롤링하려는 범위, Start time이 과거여도 가용되는 데이터부터 긁어옴            
start = dt.datetime(2021, 1, 1)
end = dt.datetime.today()

DATA = pdr.naver.NaverDailyReader("005930", start, end)
print(DATA)

# # 랑 난 국장은 모르니까 삼전이랑 네이버 찾아서 넣음, 너가 하는거 우르륵 
# # 넌 이거만 고치면 돼
# Stock_Ticker = ["005930.KS","035420.KS", "305720.KS"]
# # yahoo finance에서 데이터를 긁어옴 interval='m'이면 월변화를 가져옴
# RAW_DATA = pdr.get_data_yahoo(Stock_Ticker, start, end, interval='m')
# # 불러온 거에서 종가(Close) 값만 불러옴
# DP = RAW_DATA.Close.copy()
# # 데이터 함 맞나 보시고
# print(DP)
# # 마지막 데이터가 해당 달 마지막 날인지 보시고 지울지 말지 판단하는겨
# def yes_or_no(question):
#     while "the answer is invalid":
#         reply = str(input(question+' (y/n): ')).lower().strip()
#         if reply[0] == 'y':
#             return True
#         if reply[0] == 'n':
#             return False

# Ans = yes_or_no("마지막 꺼 잘봐 지울겨?")   
# if Ans :
#     DP.drop(DP.index[-1], inplace=True)
# print("마지막꺼 지웠다.")
# print(DP.tail())

# # 자 이제 1/3/6 모멘텀 계산하자
# A = 1/3 * (DP.iloc[-1,:] - DP.iloc[-2,:])/DP.iloc[-2,:]
# B = 1/3 * (DP.iloc[-1,:] - DP.iloc[-4,:])/DP.iloc[-4,:]
# C = 1/3 * (DP.iloc[-1,:] - DP.iloc[-7,:])/DP.iloc[-7,:]

# print((A+B+C)*100)