import os
from typing import *
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from .nlp_utils import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.utils.math import cosine_similarity
from langchain.schema import Document
from langchain.chat_models import ChatGooglePalm
from langchain.document_loaders import PyPDFLoader


class ResumeGrader:
    def __init__(self, job_description: str, resume_dir: str):
        self.jd = job_description
        self.llm = ChatGooglePalm(temperature=0.1)

        self.__load_resumes(resume_dir)

        summarization_prompt_template = """
        Write a brief summary of the following content extracted from a resume. Be sure to keep important keywords such as those pertaining to their skills, projects completed, work experience, etc.

        {text}

        SUMMARY:
        """

        prompt = PromptTemplate(
            template=summarization_prompt_template, input_variables=["text"]
        )
        self.summary_chain = load_summarize_chain(
            llm=self.llm, chain_type="stuff", prompt=prompt
        )

    def __load_resumes(self, resume_dir):
        self.resumes = dict(
            (file, PyPDFLoader(file_path=os.path.join(resume_dir, file)).load())
            for file in os.listdir(os.path.join(os.path.abspath(resume_dir)))
        )

    def grade(self):
        self.jd_summary = self.__summarize([Document(page_content=self.jd)])
        self.resume_summaries = {
            key: preprocess(self.__summarize(value))
            for key, value in self.resumes.items()
        }

        # initialize vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        vectorizer.fit([*self.resume_summaries.values(), self.jd_summary])

        jd_vector = vectorizer.transform([self.jd_summary]).toarray()
        # vectorize all summaries
        self.similarities = dict()
        for resume_key, resume_summary in self.resume_summaries.items():
            resume_vector = vectorizer.transform([resume_summary]).toarray()
            similarity = (
                1 + cosine_similarity(resume_vector, jd_vector).flatten()[0]
            ) / 2
            self.similarities[resume_key] = similarity

        return self.similarities

    def __summarize(self, paragraph: List[Document]) -> str:
        return self.summary_chain.run(paragraph)