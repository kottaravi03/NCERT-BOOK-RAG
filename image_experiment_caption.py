"""This module will containe the code for image captioning
"""
from langchain_core.messages import HumanMessage, SystemMessage
from utils import get_model_from_gcp, image_to_base64
from langchain_core.prompts import ChatPromptTemplate

llm = get_model_from_gcp()

# To add image to the prompt we need to send the base64 encoded version of bytes

image = image_to_base64("images/page_001_image_001.png")



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert in captioning images extracted from NCERT textbooks.
Use the page content to understand the context of the image.
Generate educational captions suitable for the student's grade level.
"""
        ),
        (
            "human",
            [
                {
                    "type": "text",
                    "text": """
Subject: {subject}

Standard: {standard}

Page Content:
{content}

Generate a caption for the attached image.
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,{image}"
                    }
                }
            ]
        )
    ]
)


chain = prompt | llm

result = chain.invoke({
    "image": image,
    "subject": "science",
    "standard": "7",
    "content": """7.1 What is Probability?
Probability is a type of measurement, similar to how we measure quantities like length, area, or volume. However, instead of measuring physical quantities, probability is used to measure the likelihood of events. Specifically, it helps us express how confident or certain we are that a particular event will occur. For example, you may ask your friend: 
•	Is it going to rain today?
•	Will our school win the inter-school  hockey match tomorrow? 
•	Will I be chosen in the monthly lucky draw to perform at the school assembly? [The names of all the students in school are written on slips of paper and one slip is randomly selected.]
These events are examples of random events. We know the possible outcomes 
(either it will rain today or it will not; our school team will either win, draw or lose the hockey 
match; one student will be chosen to perform at the school assembly), but we do not know in advance which one will definitely occur. That is, there is an element of chance or randomness involved every time such an event takes place. 
Can we predict these outcomes with 100% certainty? One could respond to these questions with words such as impossible or certain, or using phrases such as less likely, more likely or equally likely. This decision is based on different kinds of evidence that have been gathered. For example, one friend might say, “The sun is shining brightly, so it’s unlikely to rain today”, while another might observe, “It’s very hot, which makes me think it could rain later”. In both cases, they are predicting rainfall based on how they interpret the present weather conditions. This is the subjective probability given to the event of today’s rainfall. 
As you can see from the above example, probability deals with uncertainty or chance. One key feature of our increasingly complicated society is that we must deal with questions that have no fixed answer but rather one or more possibilities for the answer. Thus, understanding how to objectively estimate the probability of events is crucial in many aspects of life.
"""
})
print(result.content)