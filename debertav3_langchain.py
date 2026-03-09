from huggingface_hub import login
# login token here 
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
import os
import json
from typing import List, Dict, Tuple

# LangChain imports
from langchain_community.vectorstores import Redis as LangChainRedis
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.documents import Document as LCDocument

warnings.filterwarnings("ignore")

# 1. CONFIGURATION
class Config:
    # Model settings
    MODEL_NAME = 'microsoft/deberta-v3-base'
    EXPLANATION_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE_BERT = 1e-5
    LEARNING_RATE_CLF = 1e-4
    METADATA_DIM = 0
    NUM_CLASSES = 2
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # RAG settings - LangChain
    REDIS_URL = 'redis://localhost:6379'
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    TOP_K_RETRIEVAL = 3
    INDEX_NAME = 'fact_check_langchain'
    
    # Paths
    ARTICLES_PATH = 'fact_check_articles_averitec.csv'


# 2. RAG PIPELINE - LANGCHAIN VERSION
class LangChainRAGPipeline:
    
    def __init__(self, redis_url: str, embedding_model: str, index_name: str):
        print(f"Initializing RAG pipeline")
        try:
            import redis
            from redis.commands.search.field import VectorField, TextField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.index_name = index_name
            self.vector_dim = 384  # all-MiniLM-L6-v2 output dim
            print(f"Connected to Redis at {redis_url}")
            self.available = True
            
        except Exception as e:
            print(f"Could not initialize RAG: {e}")
            print("RAG features will be disabled")
            self.available = False

    def _create_index(self):
        from redis.commands.search.field import VectorField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        try:
            self.redis_client.ft(self.index_name).info()
            print("Index already exists")
        except:
            schema = [
                TextField("title"),
                TextField("text"),
                TextField("verdict"),
                VectorField("embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            ]
            definition = IndexDefinition(prefix=[f"{self.index_name}:"], index_type=IndexType.HASH)
            self.redis_client.ft(self.index_name).create_index(schema, definition=definition)
            print("Created Redis index")

    def index_articles(self, articles: List[Dict]):
        if not self.available:
            return
        
        print(f"\nIndexing {len(articles)} articles")
        self._create_index()
        
        import numpy as np
        texts = [f"{a['title']}\n\n{a['text']}" for a in articles]
        
        # Embed in batches
        batch_size = 64
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            embs = self.embeddings.embed_documents(batch)
            all_embeddings.extend(embs)
        
        # Store in Redis
        pipe = self.redis_client.pipeline()
        for i, (article, embedding) in enumerate(tqdm(zip(articles, all_embeddings), desc="Storing", total=len(articles))):
            key = f"{self.index_name}:{article['id']}"
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            pipe.hset(key, mapping={
                "title": article['title'][:200],
                "text": article['text'][:500],
                "verdict": article.get('verdict', ''),
                "id": article['id'],
                "embedding": embedding_bytes
            })
            if i % 100 == 0:
                pipe.execute()
                pipe = self.redis_client.pipeline()
        pipe.execute()
        print(f"Indexed {len(articles)} articles")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.available:
            return []
        
        try:
            import numpy as np
            from redis.commands.search.query import Query
            
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
            
            q = (
                Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
                .sort_by("score")
                .return_fields("title", "text", "verdict", "id", "score")
                .paging(0, top_k)
                .dialect(2)
            )
            
            results = self.redis_client.ft(self.index_name).search(
                q, query_params={"vec": query_vector}
            )
            
            formatted = []
            for doc in results.docs:
                formatted.append({
                    'text': getattr(doc, 'text', ''),
                    'title': getattr(doc, 'title', ''),
                    'verdict': getattr(doc, 'verdict', ''),
                    'id': getattr(doc, 'id', ''),
                    'similarity': 1 - float(getattr(doc, 'score', 1))
                })
            return formatted
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def clear_index(self):
        if not self.available:
            return
        try:
            self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
            print(f"Cleared index: {self.index_name}")
        except:
            print("Index didn't exist yet, skipping clear")


# 3. DATA PREPARATION
def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file at: {filepath}")
        
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state', 'party', 'barely_true_ct', 'false_ct', 'half_true_ct', 
        'mostly_true_ct', 'pants_fire_ct', 'context'
    ]
    df = pd.read_csv(filepath, sep='\t', header=None, names=columns)
    
    count_cols = ['barely_true_ct', 'false_ct', 'half_true_ct', 'mostly_true_ct', 'pants_fire_ct']
    df[count_cols] = df[count_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def load_fact_check_articles(filepath: str) -> List[Dict]:
    if not os.path.exists(filepath):
        print(f"Warning: Articles file not found at {filepath}")
        return []
    
    df = pd.read_csv(filepath)
    articles = []
    for _, row in df.iterrows():
        articles.append({
            'id': str(row.get('id', '')),
            'title': str(row.get('title', '')),
            'text': str(row.get('text', '')),
            'url': str(row.get('url', '')),
            'verdict': str(row.get('verdict', ''))
        })
    return articles


def convert_to_binary(label):
    reliable = ['true', 'mostly-true', 'half-true']
    return 1 if label in reliable else 0


def preprocess_features_with_langchain(df: pd.DataFrame, tokenizer, rag_pipeline: LangChainRAGPipeline) -> Tuple:
    count_cols = ['barely_true_ct', 'false_ct', 'half_true_ct', 'mostly_true_ct', 'pants_fire_ct']
    metadata = df[count_cols].apply(np.log1p)
    
    total_statements = df[count_cols].sum(axis=1)
    metadata['log_total_statements'] = np.log1p(total_statements)
    
    def map_party(p):
        if p == 'democrat': return 0
        if p == 'republican': return 1
        return 2
    
    metadata['party_enc'] = df['party'].apply(map_party)
    meta_features = metadata.values.astype(np.float32)
    
    sep = tokenizer.sep_token
    texts = []
    retrieved_contexts = []
    
    use_rag = rag_pipeline and rag_pipeline.available
    
    if use_rag:
        print("\nRetrieving relevant articles with LangChain")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", disable=not use_rag):
        statement = str(row['statement'])
        
        if use_rag:
            # Use LangChain's search
            retrieved_docs = rag_pipeline.search(statement, top_k=Config.TOP_K_RETRIEVAL)
            
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                context_parts.append(f"[Source {i+1}]: {doc['title'][:100]}")
            retrieved_context = " | ".join(context_parts)
            retrieved_contexts.append(retrieved_context)
        else:
            retrieved_context = "No RAG context"
            retrieved_contexts.append(retrieved_context)
        
        parts = [
            f"Statement: {statement}",
            f"Evidence: {retrieved_context}",
            f"Context: {str(row['context'])}",
            f"Subject: {str(row['subject'])}",
            f"Speaker: {str(row['speaker'])}"
        ]
        text = f" {sep} ".join([p for p in parts if p and 'nan' not in p.lower()])
        texts.append(text)
    
    return texts, meta_features, df['label'].values, retrieved_contexts


# 4. DATASET CLASS
class EnhancedMisinformationDataset(Dataset):
    def __init__(self, texts, metadata, labels, tokenizer, max_len):
        self.texts = texts
        self.metadata = metadata
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        meta = self.metadata[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(meta, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 5. MODEL ARCHITECTURE
class HybridDeBERTa(nn.Module):
    def __init__(self, base_model_name, num_classes, metadata_dim):
        super(HybridDeBERTa, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.text_hidden_size = self.bert.config.hidden_size
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        fusion_dim = self.text_hidden_size + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, metadata):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]
        meta_features = self.meta_mlp(metadata)
        combined = torch.cat((text_features, meta_features), dim=1)
        logits = self.classifier(combined)
        return logits


# 6. EXPLANATION GENERATION
class ExplanationGenerator:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = Config.EXPLANATION_MODEL
        
        print(f"\nInitializing explanation generator")
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )
            print("Explanation model loaded")
        except Exception as e:
            print(f"LLM not available: {e}")
            print("Using rule-based explanations")
            self.pipeline = None
    
    def create_prompt(self, statement: str, prediction: int, speaker: str, 
                     party: str, retrieved_context: str, speaker_history: Dict) -> str:
        label = "UNRELIABLE" if prediction == 0 else "RELIABLE"
        
        prompt = f"""Analyze this statement:

STATEMENT: "{statement}"
SPEAKER: {speaker} ({party})
PREDICTION: {label}

SPEAKER HISTORY:
- False: {speaker_history['false_ct']}
- Barely true: {speaker_history['barely_true_ct']}
- Half true: {speaker_history['half_true_ct']}
- Mostly true: {speaker_history['mostly_true_ct']}
- Pants on fire: {speaker_history['pants_fire_ct']}

RETRIEVED EVIDENCE:
{retrieved_context}

Explain in 2-3 sentences why this is {label}:"""
        
        return prompt
    
    def generate(self, statement: str, prediction: int, speaker: str, 
                party: str, retrieved_context: str, speaker_history: Dict) -> str:
        
        if self.pipeline is None:
            label = "unreliable" if prediction == 0 else "reliable"
            return (f"Based on analysis of the statement by {speaker}, combined with "
                    f"retrieved fact-checking evidence, this statement is classified as {label}. "
                    f"The model considered both content and speaker credibility.")
        
        prompt = self.create_prompt(statement, prediction, speaker, party, 
                                   retrieved_context, speaker_history)
        
        try:
            result = self.pipeline(prompt, max_new_tokens=300)[0]['generated_text']
            if "Explain in" in result:
                result = result.split("Explain in")[-1].split("sentences")[-1].strip()
            return result[:400]
        except:
            label = "unreliable" if prediction == 0 else "reliable"
            return f"Statement by {speaker} classified as {label} based on content and credibility analysis."


# 7. TRAINING FUNCTIONS
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step() 
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(dataloader), correct / total


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, metadata)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return total_loss / len(dataloader), correct / total, all_labels, all_preds


def eval_with_explanations(model, dataloader, df, contexts, explainer, device, n=10):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, metadata)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print(f"\nGenerating {n} explanations...")
    explanations = []
    indices = np.random.choice(len(df), min(n, len(df)), replace=False)
    
    for idx in tqdm(indices, desc="Explanations"):
        row = df.iloc[idx]
        pred = all_preds[idx]
        
        history = {
            'false_ct': int(row['false_ct']),
            'barely_true_ct': int(row['barely_true_ct']),
            'half_true_ct': int(row['half_true_ct']),
            'mostly_true_ct': int(row['mostly_true_ct']),
            'pants_fire_ct': int(row['pants_fire_ct'])
        }
        
        exp = explainer.generate(
            row['statement'], pred, row['speaker'], row['party'], contexts[idx], history
        )
        
        explanations.append({
            'statement': row['statement'],
            'speaker': row['speaker'],
            'true_label': 'Unreliable' if all_labels[idx] == 0 else 'Reliable',
            'predicted': 'Unreliable' if pred == 0 else 'Reliable',
            'confidence': float(all_probs[idx][pred]),
            'explanation': exp,
            'retrieved': contexts[idx][:200] + '...'
        })
    
    return all_labels, all_preds, explanations


# 8. MAIN
if __name__ == "__main__":
    print("ENHANCED MISINFORMATION DETECTION - LANGCHAIN VERSION")
    print(f"Device: {Config.DEVICE}\n")
    
    # Initialize LangChain RAG Pipeline
    print("STEP 1: LANGCHAIN RAG PIPELINE")
    
    rag_pipeline = LangChainRAGPipeline(
        redis_url=Config.REDIS_URL,
        embedding_model=Config.EMBEDDING_MODEL,
        index_name=Config.INDEX_NAME
    )
    
    if rag_pipeline.available:
        articles = load_fact_check_articles(Config.ARTICLES_PATH)
        if articles:
            print(f"\nLoaded {len(articles)} articles")
            rag_pipeline.clear_index()
            rag_pipeline.index_articles(articles)
    
    # Load Data
    print("STEP 2: LOAD DATA")
    
    train_df = load_data('train.tsv')
    valid_df = load_data('valid.tsv')
    test_df = load_data('test.tsv')
    
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")
    
    train_df['label'] = train_df['label'].apply(convert_to_binary)
    valid_df['label'] = valid_df['label'].apply(convert_to_binary)
    test_df['label'] = test_df['label'].apply(convert_to_binary)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Preprocess with LangChain
    train_texts, train_meta, train_y, train_ctx = preprocess_features_with_langchain(train_df, tokenizer, rag_pipeline)
    valid_texts, valid_meta, valid_y, valid_ctx = preprocess_features_with_langchain(valid_df, tokenizer, rag_pipeline)
    test_texts, test_meta, test_y, test_ctx = preprocess_features_with_langchain(test_df, tokenizer, rag_pipeline)
    
    print("NaN in train_meta:", np.isnan(train_meta).any())
    print("NaN in train_y:", np.isnan(train_y.astype(float)).any())
    print("Meta shape:", train_meta.shape)
    print("Meta sample:", train_meta[0])  

    Config.METADATA_DIM = train_meta.shape[1]
    
    train_ds = EnhancedMisinformationDataset(train_texts, train_meta, train_y, tokenizer, Config.MAX_LEN)
    valid_ds = EnhancedMisinformationDataset(valid_texts, valid_meta, valid_y, tokenizer, Config.MAX_LEN)
    test_ds = EnhancedMisinformationDataset(test_texts, test_meta, test_y, tokenizer, Config.MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=Config.VALID_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=Config.VALID_BATCH_SIZE)
    
    # Train Model
    print("STEP 3: TRAIN MODEL")
    
    model = HybridDeBERTa(Config.MODEL_NAME, Config.NUM_CLASSES, Config.METADATA_DIM).to(Config.DEVICE)
    
    weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(Config.DEVICE))
    

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': Config.LEARNING_RATE_BERT},
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': Config.LEARNING_RATE_CLF}
    ], eps=1e-6)

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_acc, _, _ = eval_model(model, valid_loader, criterion, Config.DEVICE)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.1%}")
        print(f"Valid: Loss={val_loss:.4f}, Acc={val_acc:.1%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_langchain.pt')
            print("Saved")
    
    # Evaluate
    print("STEP 4: EVALUATE")
    
    model.load_state_dict(torch.load('best_model_langchain.pt'))
    explainer = ExplanationGenerator()
    
    true_labels, preds, explanations = eval_with_explanations(
        model, test_loader, test_df, test_ctx, explainer, Config.DEVICE, n=5
    )
    
    acc = accuracy_score(true_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', pos_label=0)
    
    print("RESULTS")
    print(f"Accuracy: {acc:.1%}")
    print(f"Precision: {prec:.1%}")
    print(f"Recall: {rec:.1%}")
    print(f"F1: {f1:.3f}")
    
    print("\n" + classification_report(true_labels, preds, target_names=['Unreliable', 'Reliable']))
    
    
    print("COMPLETE - LANGCHAIN VERSION")
 
    print("\n SAMPLE EXPLANATIONS ")
    for i, exp in enumerate(explanations):
        print(f"\nExample {i+1}")
        print(f"Statement:   {exp['statement']}")
        print(f"Speaker:     {exp['speaker']}")
        print(f"True Label:  {exp['true_label']}")
        print(f"Predicted:   {exp['predicted']}")
        print(f"Confidence:  {exp['confidence']:.1%}")
        print(f"Evidence:    {exp['retrieved']}")
        print(f"Explanation: {exp['explanation']}")
