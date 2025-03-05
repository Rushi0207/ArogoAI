import nltk

nltk.download("punkt")

def chunk_text(text, chunk_size=200, overlap=50):
    """
    Split text into chunks of approximately `chunk_size` words with `overlap` words between chunks.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_length = len(words)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-(overlap // 10):]
            current_length = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
