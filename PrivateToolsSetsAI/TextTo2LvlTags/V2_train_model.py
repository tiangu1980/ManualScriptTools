import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df[['Uninstallreason', 'Catogorie', 'CatogorieName']].dropna()

def train_model(df):
    # 只需要编码CatogorieName，因为Catogorie是与之对应的整数
    catname_encoder = {name: idx for idx, name in enumerate(df['CatogorieName'].unique())}
    y = df['CatogorieName'].map(catname_encoder)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])
    
    model.fit(df['Uninstallreason'], y)
    
    return model, catname_encoder

def save_models(model, catname_encoder):
    joblib.dump(model, 'uninstall_reason_classifier.pkl')
    joblib.dump(catname_encoder, 'category_mapping.pkl')

if __name__ == '__main__':
    df = load_data('2_UnInstallFeedback_filtered.csv')
    model, catname_encoder = train_model(df)
    save_models(model, catname_encoder)
