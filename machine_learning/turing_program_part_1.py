import random

''''
The chatbot provides random responses based on a pre—defined list.

There is no correlation between the user's input and response.

The conversation continues until the user enters "exit" to end it.

In a hypothetical scenario, a Turing judge would know this is not a human.


'''

# random responses will come from this predefined list
responses = [
    "Hello, how can I help you?"
    "Tell me more about that. "
    "I see. Please go on."
    "Interesting! Can you elaborate?"
    "I'm not sure I understand. Could you clarify?"
    "That's fascinating. Tell me more."
]

# chatbot interface, handles the conversation with the user
def chat():
print("Part #1 - Turing Test Simulator.")
print("Where you can have a pseudo conversation with an AI. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        user_input = user_input.lower()
        
        if user_input == "exit":
            print("AI: Goodbye!")
            break

        # Simulate a random response from the machine
        ai_response = random.choice(responses)
        print(f"AI: {ai_response]")
        
if __name__ == "__main__":
    chat()
