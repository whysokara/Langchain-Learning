import warnings
warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

# 1. Define your model
class PersonalData(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person", gt=18)
    city: str = Field(description="City where the person lives")

# 2. Output parser from Pydantic
parser = PydanticOutputParser(pydantic_object=PersonalData)

# 3. PromptTemplate with format_instructions from parser
template = PromptTemplate(
    template="Extract the following information about a person from the input below:\n\n{format_instructions}\n\nInput: {input}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. LLM setup
models = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

# 5. Chain: prompt -> model -> parser
chain = template | models | parser

# 6. Run the chain with natural language input
result = chain.invoke({
    "input": "My name is John Doe. I am 30 years old and I live in New York."
})

print(result)
