from typing import *

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .models import CandidateProfile, Experience, JobRole


def summarize_resume(resume: list[Document], llm):
    pydantic_parser = PydanticOutputParser(pydantic_object=CandidateProfile)
    outputfixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    template = """### Summarize a resume

Summarize the provided resume into a brief, one-page document that highlights the candidate's key skills, experience, and qualifications. The summary should be written in a clear and concise style that is easy to read and understand.

The summary should include the following information:

* The candidate's name, contact information, and professional title
* A brief overview of the candidate's work experience, including their job titles, responsibilities, and accomplishments
* A list of the candidate's technical skills
* A list of the candidate's educational qualifications [Clearly indicate the level of degree (e.g. bachelor, master, phd)]
* A brief description of the candidate's personality and work style
{format_instructions}\n\nDocument:\n{document}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["document"],
        partial_variables={
            "format_instructions": outputfixing_parser.get_format_instructions()
        },
    )

    full_content = "\n".join(map(lambda x: x.page_content, resume))

    summary_chain = prompt | llm | outputfixing_parser
    final_output = summary_chain.invoke({"document": full_content})
    return final_output


def summarize_jd(jd: list[Document], llm):
    pydantic_parser = PydanticOutputParser(pydantic_object=JobRole)
    outputfixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    template = """As an expert in information extraction, you are tasked with extracting the following information from a job description:

- Minimum education
- Minimum years of experience
- List of technical skills

Do not give me general skills such as "machine learning frameworks" or "development tools". Be specific about the technical skills required.
{format_instructions}\n\nDocument:\n{document}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["document"],
        partial_variables={
            "format_instructions": outputfixing_parser.get_format_instructions()
        },
    )

    full_content = "\n".join(map(lambda x: x.page_content, jd))

    summary_chain = prompt | llm | outputfixing_parser
    final_output = summary_chain.invoke({"document": full_content})
    return final_output


def skill_summary(cskills: list[str], jdskills: list[str]):
    cskills = set(map(str.lower, cskills))
    jdskills = set(map(str.lower, jdskills))

    common_skills = cskills & jdskills
    return {
        "present": sorted(common_skills),
        "missing": sorted(jdskills - common_skills),
    }


def experience_summary(profile_experiences: list[Experience], jd_years: float):
    num_profile_years = 0
    for experience in profile_experiences:
        num_profile_years += (experience.end_date - experience.start_date).days / 365
    return {"resume": round(num_profile_years, 1), "job_description": float(jd_years)}


def education_summary(c_edu: str, jd_edu: str):
    c_edu = c_edu.lower().replace(".", "")
    jd_edu = jd_edu.lower().replace(".", "")

    education_order = ["bachelor", "master", "phd"]
    c_edu_found = education_order.index(
        list(filter(lambda edu: edu in c_edu, education_order))[0]
    )
    jd_edu_found = education_order.index(
        list(filter(lambda edu: edu in jd_edu, education_order))[0]
    )

    return {
        "resume": c_edu,
        "job_description": jd_edu,
        "satisfied": c_edu_found >= jd_edu_found,
    }


def generate_summary(profile: CandidateProfile, role: JobRole):
    """
    Parameters:
        profile: an instance of CandidateProfile containing information extracted from the candidate's resume
        role: an instance of JobRole containing information extracted from the job description

    Returns:
        a JSON-like object comparing each of skills, years of experience and education for the `profile` and `role`
    """

    return {
        "skill": skill_summary(profile.skills, role.skills_required),
        "experience": experience_summary(profile.experiences, role.experience),
        "education": education_summary(profile.degree, role.qualification_degree),
    }
