import os
import shutil

import git


def clone_repository(repo_url, local_path='./repo_temp'):
    """Clona o repositório do github"""
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    git.Repo.clone_from(repo_url, local_path)
    return local_path


def main():
    repo_url = 'https://github.com/unclecode/crawl4ai.git'
    repo_path = clone_repository(repo_url)


if __name__ == '__main__':
    main()
