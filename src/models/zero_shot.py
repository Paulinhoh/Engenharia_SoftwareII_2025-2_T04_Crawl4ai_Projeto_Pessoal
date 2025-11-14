from collections import Counter

from transformers import pipeline


classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


def classify_architecture(code_files, max_files=15):
    """Classifica padrões arquiteturais usando zero-shot"""

    # Padrões comuns
    architecture_patterns = [
        "MVC (Model-View-Controller)",
        "Microservices",
        "Monolithic",
        "Layered Architecture",
        "Event-Driven",
        "Repository Pattern",
        "Factory Pattern",
        "Singleton Pattern"
    ]

    results = []
    pattern_summary = Counter()

    for i, file_data in enumerate(code_files[:max_files], 1):
        # Analisar primeiros 500 caracteres
        code_snippet = file_data['content'][:500]

        classification = classifier(
            code_snippet,
            candidate_labels=architecture_patterns,
            multi_label=True
        )

        # Armazena o resultado
        top_patterns = list(zip(classification['labels'][:3], classification['scores'][:3]))

        results.append({
            'file': file_data['filename'],
            'path': file_data['path'],
            'top_patterns': top_patterns
        })

        for pattern, score in top_patterns:
            if score > 0.5:
                pattern_summary[pattern] += 1

    return {
        'model': 'BART Zero-Shot',
        'detailed_results': results,
        'pattern_summary': dict(pattern_summary),
        'total_files_analyzed': len(results)
    }
