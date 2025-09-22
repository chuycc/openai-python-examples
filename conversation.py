import os
from openai import OpenAI

def get_system_instructions() -> str:
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

    return instructions

class OpenAIChatBot:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", system_instructions: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.system_instructions = system_instructions or ""
        self.conversation = []

    def send_conversation(self) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            max_tokens=1500,
            temperature=0.7
        )
        return completion.choices[0].message.content

    def save_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})
        return self.conversation

    def request_user_input(self, prompt: str) -> str:
        return input(prompt).strip()

    def end_conversation(self):
        print("Goodbye!")

    def start_conversation(self):
        self.conversation = []
        self.save_message("system", self.system_instructions)

        while True:
            user_input = self.request_user_input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                self.end_conversation()
                break
            elif not user_input:
                print("Please enter a message.")
                continue
            
            self.save_message("user", user_input)
            ai_response = self.send_conversation()
            self.save_message("assistant", ai_response)

            print(f"AI: {ai_response}")

def main():
    system_instructions = get_system_instructions()

    chatbot = OpenAIChatBot(system_instructions = system_instructions)
    chatbot.start_conversation()

if __name__ == "__main__":
    main()