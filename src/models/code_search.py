import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel

# Configurar dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar tokenizador e modelo
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')
model.to(device)


def generate_code_embedding(code_text, max_files=20):
    """Gera embedding de código usando CodeBERT"""

    # Queries padrão
    search_queries = [
        "database connection and queries",
        "authentication and authorization",
        "error handling and logging",
        "API endpoints and routes",
        "data validation"
    ]

    # Gerar embeddings de código
    code_embeddings = []
    file_info = []

    for i, file_data in enumerate(code_text[:max_files], 1):
        code_text = file_data['content'][:512]  # Limite do CodeBERT

        # Tokenizar e gerar embedding
        inputs = tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Usar [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        code_embeddings.append(embedding)
        file_info.append({
            'filename': file_data['filename'],
            'path': file_data['path']
        })

    code_embeddings = np.array(code_embeddings)

    # Buscar por queries
    search_results = {}

    for query in search_queries:
        # Gerar embedding da query
        query_inputs = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            query_outputs = model(**query_inputs)

        query_embedding = query_outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Calcular similaridade com todos os arquivos
        similarities = cosine_similarity([query_embedding], code_embeddings)[0]

        # Top 5 resultados
        top_indices = np.argsort(similarities)[-5:][::-1]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            results.append({
                'rank': rank,
                'file': file_info[idx]['filename'],
                'path': file_info[idx]['path'],
                'score': float(score)
            })

        search_results[query] = results

    return {
        'model': 'CodeBERT Code Search',
        'embeddings': code_embeddings,
        'search_results': search_results,
        'queries_analyzed': search_queries,
        'total_files_indexed': len(code_text[:max_files])
    }
