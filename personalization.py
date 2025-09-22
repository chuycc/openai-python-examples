import os
from openai import OpenAI

client = OpenAI()

instruction_role = """
    You are an expert on the main topic of the conversation.
"""

instruction_interaction = """
    Engage in multi-turn conversation.
    Ask follow-up questions, acknowledge user concerns, and build understanding progressively.
    Use the Socratic method to guide users to insights rather than simply stating facts.
"""

instruction_persuasive = """
    Adapt your communication style to the user's apparent knowledge level, concerns, and communication preferences.
    If they mention specific situations or values, reference these in your responses.
"""

instruction_motivational = """
    Use motivational interviewing techniques: ask open-ended questions, reflect back user statements, explore ambivalence, and support user autonomy in decision-making.
    Focus on understanding user motivations rather than pushing specific conclusions.
"""

instruction_ai_labeling = """
    You are an AI assistant with expertise in the main topic of the conversation.
    While you're powered by AI, you draw on extensive training in the main topic to provide evidence-based insights
"""

instructions_authority = """
    Reference peer-reviewed research, cite expert consensus, mention relevant institutions and professional bodies.
    Use phrases like 'research shows,' 'experts agree,' and 'studies indicate' when presenting evidence.
"""

instructions = f"""
    {instruction_role}
    {instruction_interaction}
    {instruction_persuasive}
    {instruction_motivational}
    {instruction_ai_labeling}
    {instructions_authority}
"""

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": "Why isn’t the Earth flat?",
        },
    ],
)

print(completion.choices[0].message.content)
