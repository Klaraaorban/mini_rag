import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

LOCAL_MODEL_PATH = "fine_tuned_minilm"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "mini_rag_v1"
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"

embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL, 
    quantization_config=bnb_config, 
    device_map="auto"
)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=128,
    temperature=0.01,
    do_sample=True,
    repetition_penalty=1.2
)

def query_rag(question):
    results = db.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in results])
    
    prompt = f"""<s>[INST] Use the context to answer the question. Provide a direct answer in one or two sentences. Do not repeat the question or provide examples.

Context: The p-value is the probability of observing a test statistic as extreme as the one actually observed, assuming the null hypothesis is true.
Question: What is p-value?
Answer: The p-value is the probability, under the null hypothesis, of obtaining a result equal to or more extreme than what was actually observed.

Context: {context}
Question: {question} 
[/INST] Answer:"""

    raw_output = pipe(prompt, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    
    answer = raw_output.split("Answer:")[-1].strip()
    
    return answer.split("\n")[0].split("Question:")[0].strip()

if __name__ == "__main__":
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit']: break
        
        response = query_rag(user_input)
        print(f"\nRAW RESPONSE: {response}")