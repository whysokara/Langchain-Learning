from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

models = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b"
)

schema = [
    ResponseSchema(name='fact_1', description='The first fact about the topic'),
    ResponseSchema(name='fact_2', description='The second fact about the topic'),
    ResponseSchema(name='fact_3', description='The third fact about the topic')
]


parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give 3 facts about {topic} \n {format_instructions}",
    input_variables = ['topic'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)
"""
prompt = template.invoke({'topic':'black hole'})

result = models.invoke(prompt)
final_result = parser.parse(result.content)

print(final_result)
"""

chain = template | models | parser
result = chain.invoke({'topic': 'black hole'})
print(result)