import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import sklearn

st.set_page_config(layout='wide')

st.title('Strawberry fields forever')

penguins = sns.load_dataset('penguins')


with st.expander('PENGUINS DATA click to collapse'):
    st.dataframe(penguins)
    
option = st.selectbox('Select species',('Gentoo','Adelie','Chinstrap'))
    
    
col1, col2, col3 = st.columns(3)

if option == 'Gentoo':
    with col1:
        st.image('Images/Gentoo.png',caption='Gentoo')

if option == 'Adelie':
    with col2:
        st.image('Images/Adelie.png',caption='Adelie')
        
if option == 'Chinstrap':   
    with col3:
        st.image('Images/Chinstrap.png',caption='Chinstrap')
        

bill_length = st.number_input('Bill_length')
bill_depth = st.number_input('Bill_depth')
        
new_peng = pd.DataFrame({'bill_length_mm':[bill_length],'bill_depth_mm':[bill_depth]})

my_scaler = pickle.load(open('scaler.pickle','rb'))

new_penguin = my_scaler.transform(new_peng)

kmeans = pickle.load(open('kmeans.pickle','rb'))

st.write(kmeans.predict(new_penguin))


    
