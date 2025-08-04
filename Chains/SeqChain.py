from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt1 = PromptTemplate(
    template="Generarte a detailed report on the topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 bullets from the text \n {text}",
    input_variables=["text"]
)

models = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

parser = StrOutputParser()

chain = prompt1 | models | parser | prompt2 | models | parser

result = chain.invoke({'topic': 'bitcoin'})
print(result)
