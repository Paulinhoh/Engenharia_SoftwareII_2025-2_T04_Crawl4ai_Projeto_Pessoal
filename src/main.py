from models.text_embedding import calculate_similarity
from models.zero_shot import classify_architecture
from services.repository import clone_repository, extract_code_files, print_final_summary
from models.code_search import generate_code_embedding

REPO_URL = 'https://github.com/unclecode/crawl4ai.git'


def main():
    """Análise completa do repositório"""
    print(f"Analisando: {REPO_URL}")

    # Clonar repo
    print("\n1. Clonando repositório...")
    repo_path = clone_repository(REPO_URL)

    # Extrair arquivos
    print("\n2. Extraindo arquivos de código...")
    code_files = extract_code_files(repo_path)
    print(f"   ✓ Encontrados {len(code_files)} arquivos\n")

    if len(code_files) == 0:
        print("❌ Nenhum arquivo de código encontrado!")

    # Análise com os 3 modelos
    results = {}

    # MODELO 1: Zero-shot
    results['zero_shot'] = classify_architecture(code_files, len(code_files))

    # MODELO 2: Embeddings
    results['embeddings'] = calculate_similarity(code_files, len(code_files))

    # MODELO 3: CodeBERT
    results['codebert'] = generate_code_embedding(code_files, len(code_files))

    # 4. Resumo final
    print_final_summary(results)


if __name__ == '__main__':
    main()
