import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Dataset Path
DATASET_PATH = r"C:\Users\GAURANG MHATRE\Downloads\archive (1)\hotel_bookings.csv"

# Load and Preprocess Data
def load_and_preprocess_data():
    try:
        df = pd.read_csv(DATASET_PATH)
        df['children'] = df['children'].fillna(0)
        df.dropna(subset=['country'], inplace=True)
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' + 
            df['arrival_date_month'] + '-' + 
            df['arrival_date_day_of_month'].astype(str)
        )
        df['total_revenue'] = df['stays_in_weekend_nights'] * df['adr'] + df['stays_in_week_nights'] * df['adr']
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Analytics Functions
def revenue_trends(df):
    try:
        monthly_revenue = df.groupby(pd.Grouper(key='arrival_date', freq='M'))['total_revenue'].sum()
        plt.figure(figsize=(12, 6))
        monthly_revenue.plot(kind='line', title='Monthly Revenue Trends')
        plt.xlabel('Date')
        plt.ylabel('Total Revenue')
        plt.tight_layout()
        plt.savefig('revenue_trends.png')
        plt.close()
        return monthly_revenue.to_dict()
    except Exception as e:
        print(f"Error generating revenue trends: {e}")
        return {}

def cancellation_rate(df):
    try:
        total_bookings = len(df)
        cancelled_bookings = len(df[df['is_canceled'] == 1])
        return {
            'total_bookings': total_bookings,
            'cancelled_bookings': cancelled_bookings,
            'cancellation_rate': (cancelled_bookings / total_bookings) * 100
        }
    except Exception as e:
        print(f"Error calculating cancellation rate: {e}")
        return {}

def geographical_distribution(df):
    try:
        country_bookings = df['country'].value_counts()
        plt.figure(figsize=(15, 7))
        country_bookings.head(10).plot(kind='bar', title='Top 10 Countries by Bookings')
        plt.xlabel('Country')
        plt.ylabel('Number of Bookings')
        plt.tight_layout()
        plt.savefig('geographical_distribution.png')
        plt.close()
        return country_bookings.head(10).to_dict()
    except Exception as e:
        print(f"Error generating geographical distribution: {e}")
        return {}

def booking_lead_time_distribution(df):
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['lead_time'], bins=30, kde=True)
        plt.title('Booking Lead Time Distribution')
        plt.xlabel('Lead Time (Days)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('lead_time_distribution.png')
        plt.close()
        return {
            'mean_lead_time': df['lead_time'].mean(),
            'median_lead_time': df['lead_time'].median(),
            'max_lead_time': df['lead_time'].max()
        }
    except Exception as e:
        print(f"Error in lead time distribution: {e}")
        return {}

# RAG System Setup
def setup_vector_database(df):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        text_data = df.apply(lambda row: f"Booking in {row['country']} with {row['adults']} adults and {row['children']} children. Total revenue: {row['total_revenue']:.2f}. Lead time: {row['lead_time']} days.", axis=1)
        embeddings = model.encode(text_data.tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print("Vector database initialized!")
        return index, text_data, model
    except Exception as e:
        print(f"Error setting up vector database: {e}")
        return None, None, None

# Load Data and Setup Vector Database
df = load_and_preprocess_data()
if not df.empty:
    vector_index, text_data, embedding_model = setup_vector_database(df)
else:
    vector_index, text_data, embedding_model = None, None, None

# API Endpoints
@app.route('/analytics', methods=['POST'])
def get_analytics():
    if df.empty:
        return jsonify({'error': 'Data not loaded properly'}), 500
    analytics = {
        'revenue_trends': revenue_trends(df),
        'cancellation_rate': cancellation_rate(df),
        'geographical_distribution': geographical_distribution(df),
        'lead_time_distribution': booking_lead_time_distribution(df)
    }
    return jsonify(analytics)

@app.route('/ask', methods=['POST'])
def answer_question():
    if not vector_index or not text_data or not embedding_model:
        return jsonify({'error': 'Vector database not initialized'}), 500
    query = request.json.get('question', '')
    query_embedding = embedding_model.encode([query])
    D, I = vector_index.search(query_embedding, k=5)
    context = " ".join([text_data[idx] for idx in I[0]])
    response = f"Based on the context of booking data, here's what I found: {context}"
    return jsonify({'answer': response})

@app.route('/health', methods=['GET'])
def system_health():
    health_status = {
        'database_status': 'OK' if not df.empty else 'FAILED',
        'data_records': len(df) if not df.empty else 0,
        'vector_index_size': vector_index.ntotal if vector_index else 0
    }
    return jsonify(health_status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)