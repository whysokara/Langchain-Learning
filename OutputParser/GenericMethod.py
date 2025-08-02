from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables = ["topic"]
)

prompt1 = template1.invoke({"topic":"Black Hole"})

result = model.invoke(prompt1)

template2 = PromptTemplate(
    template = "Write a 5 line summary of the report /n {report}",
    input_variables = ["report"]
)

prompt2 = template2.invoke({"report": result.content})
result1 = model.invoke(prompt2)

print(result1.content)