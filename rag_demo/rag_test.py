dataset = []
with open('cat-facts.txt', 'r') as f:
    dataset = f.readlines()
    print(f'Loaded {len(dataset)} lines from cat-facts.txt')


import ollama

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(EMBEDDING_MODEL, chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))


for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    a = 'Cats make about 100 different sounds. Dogs make only about 10.'
    b = 'A cat’s brain is biologically more similar to a human brain than it is to a dog’s. Both humans and cats have identical regions in their brains that are responsible for emotions.'
    c = 'Cats are North America’s most popular pets: there are 73 million cats compared to 63 million dogs. Over 30% of households in North America own a cat.'

    if  a in chunk:
        print(f'Found target chunk at index {i}')
        if a.strip().casefold() == chunk.strip().casefold():
            print(f'Exact match for target chunk at index {i}')
    if b in chunk:
        print(f'Found target chunk at index {i}')
        if b.strip().casefold() == chunk.strip().casefold():
            print(f'Exact match for target chunk at index {i}')
    if c in chunk:
        print(f'Found target chunk at index {i}')
        if c.strip().casefold() == chunk.strip().casefold():
            print(f'Exact match for target chunk at index {i}')


    if (i + 1) % 100 == 0:
        print(f'{chunk}')
        print(f'Added {i + 1} chunks to vector database')

# compute cosine similarity between two vectors
def cosine_similarity(a, b):
    import math
    a = list(a)   # ensure length checks and allow multiple passes
    b = list(b)
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0  # or raise ValueError("zero-length vector")
    return dot / (norm_a * norm_b)

# retrieve top N similar chunks from the vector database
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(EMBEDDING_MODEL, query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

input_query = input('Ask me a question:')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, score in retrieved_knowledge:
    print(f' - (similarity: {score: .2f}) {chunk}')

instruction_prompt = f''' You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join(f'- {chunk}' for chunk, similarity in retrieved_knowledge)}
'''

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)


# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

