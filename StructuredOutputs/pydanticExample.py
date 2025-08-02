from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str ="kara"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="decimal value showing cgpa")


new_student = {'age' : '28', 'email':'abc@gmail.com','cgpa': 9.2}
student = Student(**new_student)

# student_dict = dict(student)
print(student.model_dump())

