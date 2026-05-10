import streamlit as st
import sqlite3
import pandas as pd
import json
import os

st.set_page_config(page_title="BharatEval Leaderboard", layout="wide")

st.title("🇮🇳 BharatEval Leaderboard")
st.markdown("The first rigorous open-source evaluation suite for Indian language LLMs.")

db_path = "data/results_database.db"

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM evaluation_results", conn)
        conn.close()
        
        st.header("Overall Scores")
        leaderboard_df = df.groupby("model").agg({
            "composite_score": "mean",
            "semantic_score": "mean",
            "language_score": "mean",
            "cultural_score": "mean",
            "is_hallucination": "mean",
        }).reset_index()
        
        leaderboard_df = leaderboard_df.sort_values(by="composite_score", ascending=False)
        st.dataframe(leaderboard_df, use_container_width=True)
        
        st.header("Raw Results")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading database: {e}")
else:
    st.info("No evaluation results found yet. Run the pipeline to generate data.")
    
leaderboard_path = "data/leaderboard.json"
if os.path.exists(leaderboard_path):
    with open(leaderboard_path, "r") as f:
        try:
            data = json.load(f)
            st.header("Insights")
            st.json(data)
        except json.JSONDecodeError:
            pass
