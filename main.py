#This code is the main code that will call the AI Model

#Importing Modules
# importing libraries
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import openpyxl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
import sys
import json

#TODO:(Developer) Change the path based on system
system_val="Server"  

if system_val=="Server":
    input_string=sys.argv[1]
    input_data=json.loads(input_string)
elif system_val=="System":
    input_data={
        "call_id": "101",
        "company_name": "BluJay Solutions",
        "company_description": "BluJay Solutions provides transportation management and global trade network solutions.",
        "customer": {
            "offerings": "Their software optimizes logistics and customs processes, ensuring smooth and compliant cross-border operations.",
            "icp": {
            "target_industry": "Manufacturing, Retail, Logistics",
            "employee_count": "15000",
            "region": "India",
            "roles": "Manager",
            "min_pricing": "301847"
            }
        },
        "vendor": {
            "requirements": "Global Trade and Supply Chain Management Solutions",
            "ivp": {
            "vendor_industry": "Transportation, Logistics, Retail",
            "clients_count": "100",
            "region": "India",
            "max_pricing": "500000",
            "year_of_establishment": "3"
            }
        }
        }


if system_val=="Server":
    base_path=r"C:\Users\chira\OneDrive\Desktop\python-shell"
elif system_val=="System":
    base_path=r"C:\Users\chira\OneDrive\Desktop\python-shell"


def initial_preprocessing(apply_df):
    r=Rake()
    l=[]
    for i in apply_df['Business Requirements']:
        r.extract_keywords_from_text(i)
        keyword_extracted = r.get_ranked_phrases()
        out=''
        for j in keyword_extracted:
            out+=j
            out+=', '
            
        l.append(out)

    apply_df['Business Requirements']=l
    
    l=[]
    for i in apply_df['Vendor Industry']:
        r.extract_keywords_from_text(i)
        keyword_extracted = r.get_ranked_phrases()
        out=''
        for j in keyword_extracted:
            out+=j
            out+=', '
            
        l.append(out)

    apply_df['Vendor Industry']=l
    
    l=[]
    for i in apply_df['What it solves']:
        r.extract_keywords_from_text(i)
        keyword_extracted = r.get_ranked_phrases()
        out=''
        for j in keyword_extracted:
            out+=j
            out+=', '
            
        l.append(out)

    apply_df['What it solves']=l
    
    l=[]
    for i in apply_df['Target Industry']:
        r.extract_keywords_from_text(i)
        keyword_extracted = r.get_ranked_phrases()
        out=''
        for j in keyword_extracted:
            out+=j
            out+=', '
            
        l.append(out)

    apply_df['Target Industry']=l
    
    return apply_df

def load_recommend(apply_df,id,features,n_recommendations=7):
    if id=='100':
        model_path=base_path+"\\client_model.pkl"
    elif id=='101':
        model_path=base_path+"\\vendor_model.pkl"
        
    with open(model_path, 'rb') as file:
        tfidf_vectorizer, cosine_sim, df = pickle.load(file)
    
    input_df=apply_df
    
    # Combine input features into a single string
    input_df["Combined Features"] = input_df[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    # Compute the TF-IDF matrix for input data
    input_tfidf_matrix = tfidf_vectorizer.transform(input_df["Combined Features"])
    
    # Compute the cosine similarity between input and existing businesses
    cosine_sim_input = linear_kernel(input_tfidf_matrix, tfidf_vectorizer.transform(df["Combined Features"]))

    # Get the indices of businesses with highest similarity to the input
    sim_scores = list(enumerate(cosine_sim_input[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    business_indices = [i for i, _ in sim_scores]
    sim_score2=[j for _,j in sim_scores]

    # Get top N recommendations
    recommendations = df.iloc[business_indices[1:n_recommendations + 1]]  # Exclude the first one (input itself)
    sim_ss = df.iloc[sim_score2[1:n_recommendations + 1]]  # Exclude the first one (input itself)
    #print(recommendations)
    final_output=[]
    final_output.append(sim_ss)
    final_output.append(recommendations)
    return final_output





if __name__=="__main__":
    # print(input_data, type(input_data))
    if input_data["call_id"]=="100":
        # Features to consider for content-based filtering
        features = [
            "Business Requirements",
            "Vendor Industry",
        ]
        
    elif input_data["call_id"]=='101':
        features = [
            "What it solves",
            "Target Industry",
        ]
        
    apply_data= {
        "Company Name":input_data['company_name'],
        "Company Description":input_data['company_description'],
        "Business Requirements": input_data['vendor']['requirements'],
        "Vendor Industry": input_data['vendor']['ivp']['vendor_industry'],
        "What it solves": input_data['customer']['offerings'],
        "Target Industry": input_data['customer']['icp']['target_industry']
        }
    
    #Pre-Processing the Input Data
    apply_df=pd.DataFrame(apply_data, index=[0])
    apply_df.head()
    
    apply_df=initial_preprocessing(apply_df)
    
    id=input_data['call_id']
    recommend = load_recommend(apply_df,id,features)
    
    recommended =pd.DataFrame(recommend[1])

    #Do Filtering based on Pricing

    if input_data['call_id']=='100':
        cus_price = input_data['customer']['icp']['min_pricing']
        recommended_range = recommended[recommended['IVP Range'] >= int(cus_price)]
    elif input_data['call_id']=='101':
        cus_price = input_data['vendor']['ivp']['max_pricing']
        recommended_range = recommended[recommended['ICP Range'] <= int(cus_price)]
    #print(recommended)
    
    if len(recommended_range)<5:
        final_recomend = recommended
    else:
        final_recomend=recommended_range
    
    # Convert DataFrame to the desired dictionary format
    result_dict = [{'company_name': name, 'company_description': des} for name, des in zip(final_recomend['Company Name'], final_recomend['Company Description'])]

    #print(result_dict)
    
    response_data={
        "call_id": id,
        "recommendations": result_dict
    }
    
    print(response_data)
    
    
    
    

