import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def calculate_similarity(code_files, max_files=30):
    """Calcula similaridade entre arquivos de código"""

    # Gerando embeddings
    embeddings = []
    file_names = []

    for i, file_data in enumerate(code_files[:max_files], 1):
        # Usar primeiros 1000 caracteres para eficiência
        text = file_data['content'][:1000]

        emb = model.encode(text)
        embeddings.append(emb)
        file_names.append(file_data['filename'])

    embeddings = np.array(embeddings)

    # Calcular matriz de similaridades
    similarity_matrix = cosine_similarity(embeddings)

    similar_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            similarity = similarity_matrix[i][j]
            if similarity > 0.5:  # Threshold
                similar_pairs.append({
                    'file1': file_names[i],
                    'file2': file_names[j],
                    'similarity': similarity
                })

    # Ordenar por similaridade
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # Estatísticas
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    max_similarity = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])

    return {
        'model': 'Sentence Transformer Embeddings',
        'embeddings': embeddings,
        'similarity_matrix': similarity_matrix,
        'similar_pairs': similar_pairs[:20],  # Top 20
        'statistics': {
            'avg_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'num_similar_pairs': len(similar_pairs)
        }
    }
