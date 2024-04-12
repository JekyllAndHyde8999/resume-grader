import os
from typing import *

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from .llms import load_llm
from .utils import generate_summary, summarize_jd, summarize_resume


class JobDescriptionGrader:
    def __init__(
        self,
        job_description_dir: str,
        resume_path: str,
        llmconfig: Dict[str, Any],
    ):
        """
        Parameters:
            job_description: a string containing the job description
            resume_dir: the path to the folder containing the resumes in pdf format
            llmconfig: a dictionary containing configuration parameters for the llm
        """

        self.resume_path = resume_path
        self.__load_jds(job_description_dir)
        self.llm = load_llm(llmconfig)

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

        self.resume_summary = summarize_resume(
            PyPDFLoader(os.path.abspath(self.resume_path)).load(), llm=self.llm
        )
        self.summaries = {
            key: generate_summary(self.resume_summary, summarize_jd(value, self.llm))
            for key, value in self.jds.items()
        }

        return self.summaries
