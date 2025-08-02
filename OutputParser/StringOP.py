from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables = ["topic"]
)


template2 = PromptTemplate(
    template = "Write a 5 line summary of the following text /n {report}",
    input_variables = ["report"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Black Hole'})
print(result)