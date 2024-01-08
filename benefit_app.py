import streamlit as st

import pandas as pd

import joblib

import re
from unidecode import unidecode 

st.set_page_config(layout="centered")

st.title('BENEFITID Tahminleme Platformu')

st.markdown('''Tahminlemenin yapilabilmesi icin Excel dosyasindaki sutunlar tam olarak asagidaki gibi olmalidir (case sensitive):
            
    - EVENT_AGE : int (EVENTDATE - BIRTHDATE) 
    - GENDER : string (GENDERDESC)  
    - CLAIMBRANCHDESC : string         
    - PAID_AMT : float    
    - COMPLAINTS : string
    - DIAGNOSISDESC : string
    - TREATMENTCODEDESC: string
    '''       
    )

@st.cache_resource
def load_xgb_model():
    return joblib.load('comp4_xgb_model.pkl')

@st.cache_resource
def load_encoders():
    return joblib.load('encoders.pkl')

@st.cache_data
def load_stopwords():
    return pd.read_excel('turkish_stopwords.xlsx')

uploaded_file = st.file_uploader('Excel dosyasini buraya yukle', type=["xls", "xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    df = df.drop(columns = 'Unnamed: 0')

    columns = ['EVENT_AGE', 'GENDER', 'CLAIMBRANCHDESC', 'PAID_AMT', 'COMPLAINTS', 'DIAGNOSISDESC', 'TREATMENTCODEDESC']

    columns_not_found = [col for col in columns if col not in df.columns]

    if columns_not_found:
        st.write(f'Sütunlar {columns_not_found} dosyada bulunamadı')
    else:
        st.dataframe(df)

    if not columns_not_found:
            
        with st.spinner("Tahminler Yapılıyor"):


            #############################################
            ################# Data Prep #################
            #############################################

            df_main = df.copy()


            df['EVENT_AGE'] = df['EVENT_AGE'].astype(int)
            df['GENDER'] = df['GENDER'].astype(object)
            df['CLAIMBRANCHDESC'] = df['CLAIMBRANCHDESC'].astype(str).fillna('')
            df['PAID_AMT'] = df['PAID_AMT'].astype(float).fillna('')
            df['COMPLAINTS'] = df['COMPLAINTS'].astype(str).fillna('')
            df['DIAGNOSISDESC'] = df['DIAGNOSISDESC'].astype(str).fillna('')
            df['TREATMENTCODEDESC'] = df['TREATMENTCODEDESC'].astype(str).fillna('')


            ####################################################################
            ############## Text, Numerical and Categorical Columns #############
            ####################################################################

            stopwords_df = load_stopwords()
            tfidf_vectorizer, scaler, encoded_labels_dict, benefit_encoded_labels_dict = load_encoders()


            ##### Text #####

            df['text'] = df["CLAIMBRANCHDESC"] + ' ' + df['COMPLAINTS'] + ' ' + df['DIAGNOSISDESC'] + ' ' + df['TREATMENTCODEDESC']

            df = df.drop(columns = ["CLAIMBRANCHDESC", "COMPLAINTS", "DIAGNOSISDESC", "TREATMENTCODEDESC"])

            stopwords_list = stopwords_df['Turkish Stopwords'].tolist()

            df['text'] = df['text'].apply(lambda text: ' '.join([word for word in text.split() if word.lower() not in stopwords_list])) # Turkce stopwordleri kaldirma
            df['text'] = df['text'].apply(lambda text: ' '.join(text.lower().split()))  # harfleri kucultme
            df['text'] = df['text'].apply(lambda text: re.sub(r'[^\w\s]', '', text)) # ozel karakterlerden kurtulma
            df['text'] = df['text'].apply(lambda text: unidecode(text)) 

            df_text = df[["text"]]
            df = df.drop(columns = "text")

            df_text = tfidf_vectorizer.transform(df_text['text'])
            df_text2 = pd.DataFrame(df_text.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

            df = df.reset_index(drop = True)
            df_text2 = df_text2.reset_index(drop = True)
            df = pd.concat([df, df_text2], axis=1)


            ##### Numerical #####

            num_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()

            df[num_cols] = scaler.fit_transform(df[num_cols])

            ##### Categorical #####

            cat_cols = df.select_dtypes(include='object').columns.tolist()

            for col in cat_cols:
                col_dict = encoded_labels_dict[col]
                df[col] = df[col].map(col_dict).astype('category')


            ####################################################
            ################# Model Prediction #################
            ####################################################
                
            xgb_model = load_xgb_model()

            preds =  xgb_model.predict(df)

            reversed_dict = {val:key for key, val in benefit_encoded_labels_dict.items()}

            df_main["PREDICTED_BENEFITID_ENCODED"] = preds
            df_main['PREDICTED_BENEFITID'] = df_main['PREDICTED_BENEFITID_ENCODED'].map(reversed_dict)
            df_main['PREDICTED_BENEFITID'] =  df_main['PREDICTED_BENEFITID'].astype(int).astype(str)

            df_main = df_main.drop(columns = "PREDICTED_BENEFITID_ENCODED" )

            st.success("Tahminler Oluşturuldu")

        st.dataframe(df_main)