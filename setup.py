from setuptools import find_packages, setup

setup(
    name="resumegrader",
    packages=find_packages(),
    version="0.1.3",
    description="A library that uses LLMs to rank resumes for a job description",
    author="Me",
    install_requires=[
        "scikit-learn",
        "langchain",
        "langchain-community",
        "nltk",
        "spacy",
    ],
)
