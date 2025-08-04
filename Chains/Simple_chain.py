from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables =['topic']
)


models = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

parser = StrOutputParser()

chain = prompt | models | parser    

result = chain.invoke({'topic' : 'langchain'})

print(result)
chain.get_graph().print_ascii()