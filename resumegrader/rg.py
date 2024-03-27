import os
from typing import *

from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

from .utils import generate_summary, summarize_jd, summarize_resume


class ResumeGrader:
    def __init__(
        self,
        job_description: str,
        resume_dir: str,
        llmconfig: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
            job_description: a string containing the job description
            resume_dir: the path to the folder containing the resumes in pdf format
            llmconfig: a dictionary containing configuration parameters for the llm
        """

        self.jd = job_description
        if llmconfig:
            self.llm = ChatOpenAI(**llmconfig)
        else:
            self.llm = ChatOpenAI(temperature=0)

        self.__load_resumes(resume_dir)

    def __load_resumes(self, resume_dir: str):
        """
        Uses langchain's PyPDFLoader to load each of the resumes into a dictionary with the filename as the key

        Parameters:
            resume_dir: the path to the folder containing the resumes in pdf format
        """
        self.resumes = dict(
            (file, PyPDFLoader(file_path=os.path.join(resume_dir, file)).load())
            for file in os.listdir(os.path.join(os.path.abspath(resume_dir)))
        )

    def grade(self) -> Dict[str, float]:
        """
        Extract important information from the job description and resumes and store them in pydantic models

        Returns:
            similarities: a dictionary mapping between the file name and its corresponding similarity to the given job description
        """

        self.jd_summary = summarize_jd([Document(page_content=self.jd)], llm=self.llm)
        self.summaries = {
            key: generate_summary(summarize_resume(value, self.llm), self.jd_summary)
            for key, value in self.resumes.items()
        }

        return self.summaries
