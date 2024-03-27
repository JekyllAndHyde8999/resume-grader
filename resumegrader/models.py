import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    email: str = Field(description="candidate's email address")
    phone_number: str = Field(description="candidate's phone number")
    location: str = Field(description="candidate's current location of residence")
    miscellaneous: Any = Field(
        description="other miscellaneous information that does not fit into other fields"
    )


class Experience(BaseModel):
    title: str = Field(description="title of the position held")
    company: str = Field(description="name of the workplace the position was held")
    description: str = Field(description="brief description of the position held")
    start_date: datetime.date = Field(description="start date of the work experience")
    end_date: Optional[datetime.date | Literal["Present", "present"]] = Field(
        description="end date of the work experience"
    )
    miscellaneous: Any = Field(
        description="other miscellaneous information that does not fit into other fields"
    )


class CandidateProfile(BaseModel):
    name: str = Field(
        description="name of the candidate whose resume is being analyzed"
    )
    contact_info: Optional[ContactInfo] = Field(
        description="contact info of the candidate"
    )
    skills: Optional[list[str]] = Field(
        description="list of technical skills the candidate possesses"
    )
    degree: Literal["Bachelor", "Master", "PhD"] = Field(
        description="the most recent degree listed in the resume"
    )
    education: str = Field(
        description="the area of study of the most recent education listed in the resume"
    )
    experiences: Optional[list[Experience]] = Field(
        description="list of places and other jobs the candidates had in the past"
    )
    years_of_experience: Optional[float] = Field(
        description="total number of years of experience the candidate possess; extracted from work experience in resume"
    )
    miscellaneous: Any = Field(
        description="other miscellaneous information that does not fit into other fields"
    )


class JobRole(BaseModel):
    job_title: str = Field(description="title of the job role in the job description")
    skills_required: list[str] = Field(
        description="list of technical skills required for this job; each element in this list is one skill"
    )
    qualification_degree: Literal["Bachelor", "Master", "PhD"] = Field(
        description="the minimum education degree required for the job"
    )
    qualification_major: str = Field(
        description="the area of study required for the job"
    )
    experience: float = Field(
        description="the minimum number of years of experience required for the position"
    )
    miscellaneous: Any = Field(
        description="other miscellaneous information that does not fit into other fields"
    )
