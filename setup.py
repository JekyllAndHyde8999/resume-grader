from setuptools import find_packages, setup

setup(
    name="resumegrader",
    packages=find_packages(),
    version="0.2.0",
    description="A library that uses LLMs to rank resumes for a job description",
    author="Sravan Vinakota",
    install_requires=[
        "langchain",
        "langchain-community",
        "pydantic",
        "pypdf",
    ],
)
