import os

from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.utils.math import cosine_similarity
from langchain_community.chat_models import ChatGooglePalm
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from .nlp_utils import preprocess


class JobDescriptionGrader:
    def __init__(self, job_description_dir: str, resume_path: str, llmconfig: Optional[Dict[str, Any]]=None):
        """
        Parameters:
            job_description: a string containing the job description
            resume_dir: the path to the folder containing the resumes in pdf format
            llmconfig: a dictionary containing configuration parameters for the llm
        """

        self.resume_path = resume_path
        if llmconfig:
            self.llm = ChatGooglePalm(**llmconfig)
        else:
            self.llm = ChatGooglePalm(temperature=0.1)

        self.__load_jds(job_description_dir)

        summarization_prompt_template = """Write a brief summary of the following content extracted from a resume. Be sure to keep important keywords such as those pertaining to their skills, projects completed, work experience, etc.
{text}

SUMMARY:
"""

        prompt = PromptTemplate(
            template=summarization_prompt_template, input_variables=["text"]
        )
        self.summary_chain = load_summarize_chain(
            llm=self.llm, chain_type="stuff", prompt=prompt
        )

    def __load_jds(self, jd_dir):
        """
        Uses langchain's PyPDFLoader to load each of the resumes into a dictionary with the filename as the key
        
        Parameters:
            resume_dir: the path to the folder containing the resumes in pdf format
        """
        self.jds = dict(
            (file, TextLoader(file_path=os.path.join(jd_dir, file)).load())
            for file in os.listdir(os.path.join(os.path.abspath(jd_dir)))
        )

    def grade(self):
        """
        Vectorize the summaries of the resume and job descriptions
        Calculate the cosine similarity between the resume and each of the job descriptions and store the results in a dictionary
        
        Returns:
            similarities: a dictionary mapping between the job description and its corresponding similarity to the given resume
        """

        self.resume_summary = self.__summarize(
            PyPDFLoader(os.path.abspath(self.resume_path)).load()
        )
        self.jd_summaries = {
            key: preprocess(self.__summarize(value)) for key, value in self.jds.items()
        }

        # initialize vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        vectorizer.fit([*self.jd_summaries.values(), self.resume_summary])

        resume_vector = vectorizer.transform([self.resume_summary]).toarray()
        # vectorize all summaries
        self.similarities = dict()
        for jd_key, jd_summary in self.jd_summaries.items():
            jd_vector = vectorizer.transform([jd_summary]).toarray()
            similarity = (
                1 + cosine_similarity(resume_vector, jd_vector).flatten()[0]
            ) / 2
            self.similarities[jd_key] = similarity

        return self.similarities

    def __summarize(self, paragraph: List[Document]) -> str:
        return self.summary_chain.run(paragraph)
