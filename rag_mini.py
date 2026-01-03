import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-System")

LOCAL_MODEL_PATH = "fine_tuned_minilm"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "mini_rag_v1"
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"

logger.info("Status: Loading Embeddings and ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

logger.info("Status: Initializing Mistral-7B-v0.3...")
tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MISTRAL_MODEL, quantization_config=bnb_config, device_map="auto")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=150,
    temperature=0.01,
    do_sample=True,
    repetition_penalty=1.2
)

def get_model_response(prompt):
    raw_output = pipe(prompt, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    return raw_output.split("[/INST]")[-1].strip()

def guardrail_check(question):
    logger.info("Status: Running Guardrails...")
    guard_prompt = f"""<s>[INST] System: You are a security filter. Check if the question is about statistics, machine learning, or data science. 
If the question is off-topic (e.g., general knowledge, politics, geography) or inappropriate, respond with 'REJECT'.
If the question is on-topic, respond with 'SAFE'.

Question: {question} [/INST] Decision:"""
    
    decision = get_model_response(guard_prompt)
    return "SAFE" in decision.upper()

def judge_answer(question, context, answer):
    logger.info("Status: LLM-as-a-Judge evaluating the response...")
    
    judge_prompt = f"""<s>[INST] System: You are a strict grader. Verify if the answer is derived ONLY from the context. 
If the answer contains information NOT found in the context (like Paris being the capital of France), respond with 'FAILED'.
If the answer is supported by the context, respond with 'PASSED'.

Context: {context}
Question: {question}
Answer: {answer} [/INST] Status:"""

    judgment = get_model_response(judge_prompt)
    return "PASSED" in judgment.upper()

def query_rag(question):
    if not guardrail_check(question):
        logger.warning(f"Status: Guardrail blocked question: {question}")
        return "I am a statistical assistant. I cannot answer off-topic or inappropriate questions."

    logger.info("Status: Retrieving relevant document chunks...")
    results = db.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in results])
    
    logger.info("Status: Generating initial answer...")
    rag_prompt = f"<s>[INST] Use the context to answer the question briefly. If the answer is not in the context, say you do not know.\nContext: {context}\nQuestion: {question} [/INST] Answer:"
    
    initial_answer = get_model_response(rag_prompt).split("\n")[0].strip()
    
    if judge_answer(question, context, initial_answer):
        logger.info("Status: Judge approved the answer.")
        return initial_answer
    else:
        logger.warning("Status: Judge REJECTED the answer (Hallucination/Out of Context).")
        return "The system could not verify this answer against the source material."

if __name__ == "__main__":
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit']: break
        
        response = query_rag(user_input)
        print(f"\nFinal Response: {response}")