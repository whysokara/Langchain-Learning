from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n {notes} and {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
text = """Blockchain also has potential applications beyond digital assets, such as bitcoin and cryptocurrency

From a business perspective, it’s helpful to think of blockchain technology as a type of next-generation business process improvement software. Collaborative technology, such as blockchain, proclaims the ability to improve the business processes that occur between companies, radically lowering the “cost of trust.” For this reason, it may offer significantly higher returns for each investment dollar spent than most traditional internal investments.

For an overview of digital assets, which include cryptocurrencies, start with Demystifying cryptocurrency and digital assets. We provide an introduction into the mechanics of the digital asset world, how it functions, the various categories of assets, and where the future of this space could lead.

For a deeper understanding of digital assets, we recommend these resources.

Digital Assets and Crypto: PwC’s open source of knowledge on all things crypto.
PwC 2023 Digital Asset Predictions provides our thoughts on where the industry is heading to help business leaders craft the strategy that this promising but volatile space requires — one that matches innovation with trust.
What’s next for these five crypto and NFT trends in 2023?
Why is crypto custody important for financial institutions? A discussion on how custody is the foundation for any business venture into digital assets.
For an overview into web3, we recommend Demystifying web3 which discusses what business leaders should know about web3, its potential, and what no regrets decisions you can make to prepare. Here are two more recommendations.

Five ways to deepen customer engagement and loyalty with web3 which discusses how loyalty programs can be improved using web3 technology.
For a deeper understanding of blockchain technology, we suggest Embracing sustainable innovation: understanding the environmental impacts of blockchain technology, which discusses in detail how blockchain technology can be used to improve sustainability strategies.

For a look into our historical thinking in this space, we recommend:

Money is no object explores the early days of bitcoin and provides survey data on consumer familiarity, usage and more. We also look at how market participants — such as investors, technology providers and financial institutions — will be affected as the market matures.
A strategist’s guide to blockchain examines the potential benefits of this important innovation and suggests a way forward for financial institutions. Explore how others might try to disrupt your business with blockchain technology and how your company could use it to leap ahead instead.
Building blocks: How financial services can create trust in blockchain discusses some of the issues internal audit and other parties may have with a blockchain solution and how you can start to overcome some of those concerns."""
result = chain.invoke({'text': text})
print(result)