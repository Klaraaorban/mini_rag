import os
import logging
import chromadb
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "mini_rag_v1"
BASE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_PATH = "fine_tuned_minilm"
DATA_DIR = "data"

PDF_FILES = [
    os.path.join(DATA_DIR, "ISLP_website.pdf")
    # os.path.join(DATA_DIR, "Machine.Learning.with.PyTorch.and.Scikit-Learn.Sebastian.Raschka.Packt.9781801819312.EBooksWorld.ir.pdf")
]

all_documents = []
for file_path in PDF_FILES:
    if os.path.exists(file_path):
        print(f"Status: Loading {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = os.path.basename(file_path)
        all_documents.extend(docs)
    else:
        print(f"Verbose: Skipping missing file {file_path}")

if not all_documents:
    print("Fatal: No source material found.")
    exit()

print("Status: Fine-tuning embedding model on domain literature...")
fine_tune_model = SentenceTransformer(BASE_EMBEDDING_MODEL)
train_texts = [doc.page_content for doc in all_documents[:300]]
train_examples = [InputExample(texts=[t, t]) for t in train_texts]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(fine_tune_model)
fine_tune_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, show_progress_bar=True)
fine_tune_model.save(LOCAL_MODEL_PATH)
print(f"Status: Local weights saved to {LOCAL_MODEL_PATH}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=150, 
    separators=["\n\n", "\n", " ", ""]
)
texts = text_splitter.split_documents(all_documents)
print(f"Status: Created {len(texts)} chunks with n-gram overlap.")

embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)

print(f"Status: Updating ChromaDB at {CHROMA_DIR}...")
db = Chroma.from_documents(
    texts, 
    embeddings, 
    persist_directory=CHROMA_DIR, 
    collection_name=COLLECTION_NAME
)
db.persist()
print("Status: Ingestion complete. System ready for local RAG.")