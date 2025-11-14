import os
import shutil

import git


def clone_repository(repo_url, local_path='../repo_temp'):
    """Clona o repositório do github"""
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    git.Repo.clone_from(repo_url, local_path)
    return local_path


def extract_code_files(repo_path, extensions=('.py', '.js', '.java', '.cpp', '.c', '.go')):
    """Extrai os arquivos do repositorio"""
    code_files = []

    for root, dirs, files in os.walk(repo_path):
        # Ignorar diretórios comuns
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', 'venv', '__pycache__']]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        code_files.append({
                            'path': file_path,
                            'filename': file,
                            'content': content
                        })
                except:
                    continue

    return code_files


def print_final_summary(results):
    """Imprime resumo consolidado dos 3 modelos"""
    print("\n\n" + "=" * 60)
    print("🎯 RESUMO FINAL - ANÁLISE DOS 3 MODELOS")
    print("=" * 60)

    print("\n┌─ MODELO 1: ZERO-SHOT CLASSIFICATION")
    print("│  Padrão mais comum:",
          list(results['zero_shot']['pattern_summary'].keys())[0]
          if results['zero_shot']['pattern_summary'] else "Nenhum")
    print(f"│  Arquivos analisados: {results['zero_shot']['total_files_analyzed']}")

    print("\n┌─ MODELO 2: TEXT EMBEDDINGS")
    print(f"│  Similaridade média: {results['embeddings']['statistics']['avg_similarity']:.2%}")
    print(f"│  Pares similares: {results['embeddings']['statistics']['num_similar_pairs']}")

    print("\n┌─ MODELO 3: CODE SEARCH")
    print(f"│  Queries executadas: {len(results['codebert']['queries_analyzed'])}")
    print(f"│  Arquivos indexados: {results['codebert']['total_files_indexed']}")

    print("\n" + "=" * 60)
    print("✅ ANÁLISE COMPLETA FINALIZADA!")
    print("=" * 60 + "\n")
