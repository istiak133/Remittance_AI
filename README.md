# Bangladesh Remittance RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying and analyzing Bangladesh remittance data. This project provides a natural language interface to historical remittance data, allowing users to ask questions about remittance trends, year-over-year changes, and specific yearly figures.

## Features

- **Natural Language Queries**: Ask questions in plain English about Bangladesh's remittance data
- **Data Retrieval**: Semantic search to find the most relevant data for your query
- **Structured Data Extraction**: Automatic extraction of answers from structured remittance data
- **Command-line Interface**: Both interactive mode and direct query support
- **Vector Indexing**: Fast retrieval using FAISS vector database

## Installation

### Prerequisites

- Python 3.8+ 
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/istiak133/Remittance_AI.git
   cd Remittance_AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-line Interface

You can use the system in two ways:

1. **Interactive Mode**:
   ```bash
   python ask_remittance.py
   ```
   Then type your questions at the prompt.

2. **Direct Query**:
   ```bash
   python ask_remittance.py "What was the remittance in 2022?"
   ```

### Example Questions

- "What was the remittance in 2020?"
- "Show me the data for 2022"
- "How did remittance change from 2020 to 2021?"
- "What was the Year-over-Year change in 2023?"
- "Which year had the highest remittance?"

### Programmatic Usage

You can also import the RAG pipeline in your own Python code:

```python
from src.rag_pipeline import RemittanceRAGPipeline

# Initialize the pipeline
rag = RemittanceRAGPipeline('data/Bangladesh Remittances Dataset (19952025).csv')

# Ask a question
answer = rag.generate("What was the remittance in 2022?")
print(answer)
```

## Project Structure

```
remittance-ai-core/
├── ask_remittance.py         # Main CLI interface
├── src/
│   ├── rag_pipeline.py       # Core RAG implementation
│   ├── data_engine.py        # Data processing utilities
│   ├── context_engine.py     # Context handling
│   └── pattern_engine.py     # Pattern matching for answers
├── data/
│   └── Bangladesh Remittances Dataset (19952025).csv  # Historical data
├── notebooks/               # Analysis notebooks
├── tests/                  # Unit tests
└── requirements.txt        # Project dependencies
```

## How It Works

1. **Data Indexing**: The system creates vector embeddings of the remittance data using Sentence Transformers.
2. **Query Processing**: User questions are converted to the same vector space.
3. **Retrieval**: Relevant data is retrieved using FAISS approximate nearest neighbor search.
4. **Answer Generation**: The system extracts answers from the structured data based on the query patterns.

## Extending the System

### Adding New Data

To update with new remittance data:

1. Add your CSV file to the `data/` directory
2. Update the file path when initializing the RAG pipeline

### Customizing the Pipeline

The core `RemittanceRAGPipeline` class can be extended or modified to:
- Support additional data formats
- Implement more complex answer extraction logic
- Integrate with other LLMs or embedding models

## License

[MIT License](LICENSE)

## Acknowledgements

- Sentence Transformers for vector embeddings
- FAISS for efficient vector search
- Bangladesh Bank for remittance data
