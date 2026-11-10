import streamlit as st
import dill
import pandas as pd
with open('gender_label.pkl','rb') as f:
  gen_le=dill.load(f)
with open('smoker_label.pkl','rb') as f:
  smoker_le=dill.load(f)
with open('ohe.pkl','rb') as f:
  ohe=dill.load(f)
with open('model.pkl','rb') as f:
  model=dill.load(f)
with open('scaler.pkl','rb') as f:
  scaler=dill.load(f)
df=pd.read_csv('insurance.csv')
sex=st.selectbox('Select your Gender :',list(df['sex'].unique()))
age=st.number_input('Enter your Age :',min_value=df['age'].min(),max_value=df['age'].max())
bmi=st.selectbox('Select your BMI :',list(df['bmi'].unique()))
children=st.slider('Select your Children :',min_value=df['children'].min(),max_value=df['children'].max(),step=1)
smoker=st.selectbox('Select your Smoking Status :',list(df['smoker'].unique()))
region=st.selectbox('Select your Region :',list(df['region'].unique()))
bmi_value=st.number_input('Enter your BMI Value :',min_value=10,max_value=55)
age_bmi=age*bmi_value
def judge_price(sex,age,bmi,children,smoker,region,age_bmi): 
  input={
      'sex':[sex],
      'age':[age],
      'bmi':[bmi],	
      'children':[children],
      'smoker':[smoker],
      'region':[region],
      'age_bmi':[age_bmi]
  }
  df=pd.DataFrame(input)
  df['sex']=gen_le.transform(df['sex'])
  df['smoker']=smoker_le.transform(df['smoker'])
  features=ohe.transform(df[['region','bmi']])
  features=pd.DataFrame(features,columns=ohe.get_feature_names_out(),index=df.index).astype('int32')
  df=pd.concat([df.drop(columns=['region','bmi']),features],axis=1)
  df[['age_bmi','children']]=scaler.transform(df[['age_bmi','children']])
  model.predict(df)[0]
if st.button('Judge Price'):
  container=st.container(border=True)  
  price=judge_price(sex,age,bmi,children,smoker,region,age_bmi)
  container.success(price)