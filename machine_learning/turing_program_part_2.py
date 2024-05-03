'''

The chatbot responds to specific keywords or phrases defined in the responses 
dictionary and provides predefined responses for those keywords.

If the user's input does not match any keywords, it returns a default response.

The conversation continues until the user enters "exit" to end it.

This is more of a "intelligent machine" where the reponses correspond to the user's question.
In a hypothetical scenario, the Turing judge would NOT know this is machine.



'''

from datetime import datetime



# dictionary (responses) contains a set of key—value pairs, chatbot responds based on the keyword detected
responses = {
    "hello" : "Hello! How can I assist you today?",
    "how are you": "I'm just a computer program, so I don't have feelings, but I'm here to help!",
    "bye": "Goodbye! Feel free to return if you have more questions .",
    "thanks": "You're welcome! Is there anything else you'd like to know?",
    "old": "I don't have an age. I'm just a program running on a computer.",
    "time": "The current date and time is: {current_date}",
}

# chatbot interface, handles the conversation with the user
def chat():
print("Part #2 — Turing Simulation Chatbot.")
print("Where you can have a pseudo conversation with an AI. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ").lower()

        if user_input == "exit ":
            print ("AI: Goodbye!")
            break
        
        ai_response = generate_response(user_input)
        print(f"AI: {ai_response}")
        
# takes user's input as a parameter and determines the appropriate response
def generate_response(user _ input):
    for keyword, response in responses.items():
        if keyword in user_input:
            if keyword == "time":
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return response.format(current_date=current_date)
            else:
                return response

    # If no keyword is found, generate a default response
    return "I'm not sure I understand. Can you please rephrase your question?"
    
if __name__ == "__main__":
    chat()
