import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
# Created one function that does all text preprocessing in one go
def transform_text(text):
    # 1.for lower case
    text=text.lower()
    # 2.Tokenization
    text=nltk.word_tokenize(text)
    # 3. empty list created to store only alpha numeric and removing special characters
    y=[] 
    for word in text:
        if word.isalnum():
            y.append(word)
    text=y[:]# cloning y in text copying y in text bcoz it is immutable thus direct assigning is not done
    y.clear()# empty the list y
    # 4.Removing stop words and punctuations
    for word in text: 
        if  word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)
    text=y[:]# cloning y in text copying y in text bcoz it is immutable thus direct assigning is not done
    y.clear()# empty the list y
    # 5. Stemming
    for word in text:
        ps=PorterStemmer()# object created
        y.append(ps.stem(word))
        
    return " ".join(y) # converting list into string 
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Classifier")
input_sms=st.text_area("Enter the message")
if st.button("Predict"):
    #1.Preprocess
    transform_sms=transform_text(input_sms)
    #2.Vectorize
    vector_input=tfidf.transform([transform_sms])
    #3.Model prediction
    result=model.predict(vector_input)[0]
    #4.Display output
    if result==0:
        st.header("Not Spam")
    else:
        st.header("Spam")
