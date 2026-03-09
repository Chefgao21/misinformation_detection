# Enhanced Misinformation Detection with RAG & LLM Explanations

Complete fake news detection system with RAG pipeline and chain-of-thought explanations.



## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Redis (Local)
```bash
# Mac
brew install redis-stack
brew services start redis-stack

# Ubuntu
sudo apt-get install redis-stack-server -y
sudo systemctl start redis-stack-server

# Verify Redis + RediSearch loaded
redis-cli ping
redis-cli module list  # should show "search" in the list
```

### 2a. Start Redis (Google Colab)
```bash
!/opt/redis-stack/bin/redis-stack-server --daemonize yes
!sleep 3
!redis-cli ping
```

### 3. Authenticate with HuggingFace (for Llama explanations)
```python
from huggingface_hub import login
login(token="your_hf_token_here")
```
Get your token at huggingface.co/settings/tokens (Read access).
Accept Meta's license at huggingface.co/meta-llama/Llama-3.2-3B-Instruct.

### 4. Prepare Data
You need:
- `train.tsv`, `valid.tsv`, `test.tsv` (LIAR dataset)
- `fact_check_articles_averitec.csv` (AVeriTeC RAG articles)

### 5. Run Training
```bash

# LangChain + Redis RAG version
python debertav3_langchain.py
```

### 6. Check Outputs
After training completes:
- `best_model_langchain.pt` — saved model checkpoint
- `explanations_langchain.json` — 5 sample predictions with explanations
- `metrics_langchain.json` — accuracy, precision, recall, F1



## Architecture

### 1. RAG Pipeline
```
Take a user statement, convert it to a numerical vector (embedding), search Redis vector database, get the 3 most similar articles, combine those articles with the original statement
```

### 2. Model
```
Combine DeBERTa text embeddings and MLP-processed metadata features into a classifier that outputs the prediction
```

### 3. Explanations
```
Feed the statement, prediction, evidence, and speaker history into Llama 3.2 to generate a natural language explanation
```


## Files Structure

```
.
├── derbtav3_langchain.py                 # Main training script with langchain
├── requirements.txt                      # Dependencies
├── README.md                             
├── train.tsv                             # Training data (LIAR)
├── valid.tsv                             # Validation data
├── test.tsv                              # Test data
├── fact_check_articles_averitec.csv      # fact checking articles (AVeriTeC)
├── rag.py                                # Preprocess AVeriTeC for RAG
├── fact_check_articles_averitec.csv      # Postprocessed AVeriTeC 

```

## Configuration

Edit `Config` class in the script:

```python
class Config:
    MODEL_NAME = 'microsoft/deberta-v3-base'
    EXPLANATION_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
    EPOCHS = 10
    BATCH_SIZE = 16
    TOP_K_RETRIEVAL = 3  # Number of articles to retrieve
```


## Implementation Details

### RAG Retrieval
- Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Cosine similarity search in Redis
- Returns top-3 most similar fact-checking articles
- Context concatenated with statement before model input

### Explanation Generation
- Llama 3.2 3B Instruct for text generation
- Structured prompt with 5-step chain-of-thought
- Includes statement, speaker history, and retrieved evidence
- Generates 2-3 sentence explanations

### Data Flow
```
1. Load LIAR statements + AVeriTeC articles
2. Index articles in Redis (one-time)
3. For each statement:
   - Retrieve top-3 similar articles
   - Concatenate with statement
   - Feed to DeBERTa model
4. Generate explanations for sample predictions
5. Save results
```


## Citation

AVeriTeC Dataset:
```
@inproceedings{schlichtkrull-etal-2023-averitec,
    title = "AVeriTeC: A Dataset for Real-world Claim Verification",
    author = "Schlichtkrull, Michael and others",
    year = "2023"
}
```

## License

MIT License - Free to use and modify

## Acknowledgments

- Microsoft for DeBERTa
- Meta for Llama models
- Sentence-Transformers library
- Redis for vector storage
- AVeriTeC dataset creators
