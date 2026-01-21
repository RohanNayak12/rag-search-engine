import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_txt(
        txt:str,
        chunksize:int=500,
        overlap:int=100,
)->list[str]:
    tokens = tokenizer.encode(txt)
    chunks = []

    start = 0
    while start < len(tokens):
        end=start+chunksize
        chunk_tokens=tokens[start:end]
        chunk_txt=tokenizer.decode(chunk_tokens)
        chunks.append(chunk_txt)
        start+=chunksize-overlap
    return chunks