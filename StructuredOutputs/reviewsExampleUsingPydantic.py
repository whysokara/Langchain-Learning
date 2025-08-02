import warnings
warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, Literal



load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

class Review(BaseModel):
    key_themes: list[str] = Field(description = "Write down all the key themes discussed in the review in list format")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="return sentiment of the review either pos or neg")
    pros: Optional[list[str]] = Field(default=None,description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None,description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the product reviewer")

structured_model = llm.with_structured_output(Review)

response = structured_model.invoke("""I've been using the Redmi Note 13 Pro for about 3 weeks now, and overall, I'm quite satisfied with the purchase.
Pros:

Performance: The Snapdragon 7 Gen 1 handles multitasking and gaming smoothly. I didn’t experience any lags even while playing BGMI or switching between apps.
Display: The AMOLED screen is vibrant with excellent brightness levels. Watching YouTube and Netflix is a treat.
Battery Life: It easily lasts a full day with moderate to heavy usage. The 67W fast charging is a lifesaver—it goes from 20% to 100% in about 45 minutes.
Camera: The 200MP main camera delivers stunning shots in daylight. Portraits come out nicely with good edge detection. Night mode is decent, though not the best in class.
Build & Design: Feels premium in hand. The matte finish on the back prevents fingerprints.
Cons:
Bloatware: Comes with some pre-installed apps that I had to uninstall manually.
Low-light camera performance: While decent, it's not on par with competitors like the Pixel 6a.
No IP rating: Would’ve been great to have some kind of water resistance.""")

dictResponse = dict(response)
# print(type(dictResponse))
print(dictResponse["pros"])


