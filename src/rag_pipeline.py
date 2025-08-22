# RAG Pipeline for Bangladesh Remittance Dataset

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from typing import List, Tuple, Dict, Any, Union

class RemittanceRAGPipeline:
    def __init__(self, csv_path: str, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.df = None
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer(self.embedding_model)
        self._load_and_preprocess()
        self._build_index()
        # Actual column names from the dataset
        self.column_names = ['Year', 'Remittances (million USD)', 'Remittances (billion BDT)', 
                           'YoY Change (%)', 'Cumulative Growth vs. 1995-1996 (%)']

    def _load_and_preprocess(self):
        self.df = pd.read_csv(self.csv_path)
        self.df.fillna('', inplace=True)
        # Store both raw dataframe rows and joined text for different retrieval methods
        self.texts = self.df.apply(lambda row: ' '.join([str(x) for x in row]), axis=1).tolist()

    def _build_index(self):
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Retrieve relevant data with both text and structured format"""
        query_emb = self.model.encode([query])
        D, I = self.index.search(np.array(query_emb), top_k)
        
        results = []
        for idx, i in enumerate(I[0]):
            # Get the raw text
            text = self.texts[i]
            # Get the original dataframe row
            row_data = self.df.iloc[i].to_dict()
            # Add both to results with the score
            results.append((text, float(D[0][idx]), row_data))
            
        return results

    def extract_direct_answer(self, query: str, retrieved_data: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """Extract direct answers from structured data based on query patterns"""
        # Extract year from query if present
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
        year = None
        if year_match:
            year = int(year_match.group(0))
            
        # Look for specific columns in query based on actual column names
        if 'remittance' in query.lower() and 'bdt' in query.lower():
            column = 'Remittances (billion BDT)'
        elif 'remittance' in query.lower():
            column = 'Remittances (million USD)'
        elif 'change' in query.lower() or 'growth' in query.lower() or 'yoy' in query.lower():
            column = 'YoY Change (%)'
        elif 'total' in query.lower() or 'cumulative' in query.lower():
            column = 'Cumulative Growth vs. 1995-1996 (%)'
        else:
            column = None
            
        # Find the most relevant row matching the year if specified
        answer = ""
        if year and column:
            for _, _, row_data in retrieved_data:
                if int(float(row_data.get('Year', 0))) == year:
                    value = row_data.get(column, None)
                    if value is not None:
                        col_name = column.split(' (')[0]  # Remove units for clean display
                        answer = f"The {col_name} for {year} was {value} Million USD."
                        break
        
        # If couldn't extract direct answer, return the most relevant row with labels
        if not answer and retrieved_data:
            _, _, top_row = retrieved_data[0]
            year_val = int(float(top_row.get('Year', 0)))
            answer = f"For year {year_val}:\n"
            for col, val in top_row.items():
                if col != 'Year' and val:  # Skip year as we already mentioned it
                    answer += f"- {col}: {val}\n"
                    
        return answer

    def generate(self, query: str, top_k: int = 5) -> str:
        """Combined approach with direct answer extraction"""
        # Get retrieved data with both text and structured format
        retrieved_data = self.retrieve(query, top_k)
        
        try:
            # First attempt direct answer extraction using the structured data
            # Extract year from query if present
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
            year = None
            if year_match:
                year = int(year_match.group(0))
            
            # Prepare answer
            answer = ""
            
            # Find data for specific year if mentioned
            if year:
                for _, _, row_data in retrieved_data:
                    row_year = row_data.get('Year')
                    # Handle both string and float year values
                    if isinstance(row_year, str) and row_year.strip() == str(year):
                        row_year = int(year)
                    elif isinstance(row_year, (int, float)):
                        row_year = int(float(row_year))
                    
                    if row_year == year:
                        answer = f"For year {year}:\n"
                        # Add all values from the row
                        for col, val in row_data.items():
                            if col != 'Year':  # Skip year as we already mentioned it
                                answer += f"- {col}: {val}\n"
                        break
            
            # If no specific year or couldn't find match, return most relevant row
            if not answer and retrieved_data:
                _, _, top_row = retrieved_data[0]
                year_val = top_row.get('Year')
                if isinstance(year_val, (int, float)):
                    year_val = int(float(year_val))
                answer = f"Most relevant data (Year {year_val}):\n"
                for col, val in top_row.items():
                    if col != 'Year':  # Skip year as we already mentioned it
                        answer += f"- {col}: {val}\n"
            
            # Return the final answer
            return answer
            
        except Exception as e:
            # Fallback if something goes wrong
            return f"Couldn't process the query properly. Here's the raw relevant information:\n{retrieved_data[0][0]}"

# Example usage:
# rag = RemittanceRAGPipeline('data/Bangladesh Remittances Dataset (19952025).csv')
# print(rag.generate('What was the remittance in 2020?'))
