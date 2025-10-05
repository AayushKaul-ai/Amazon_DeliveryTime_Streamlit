import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model_compressed.pkl")
