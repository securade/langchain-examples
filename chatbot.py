from langchain.chains.llm import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template("You said: {user_input}. Here's a response:")

# Create an instance of the OpenAI LLM
llm = OpenAI()

# Create a chatbot chain
chatbot_chain = LLMChain(llm=llm, prompt=prompt_template)

# Define a function to handle user input
def handle_user_input(user_input):
    # Create a chat prompt
    chat_prompt = prompt_template.format_prompt(user_input=user_input).to_messages()

    # Generate a response using the chatbot chain
    response = chatbot_chain.invoke(chat_prompt)

    # Return the response
    return response['text']

# Test the chatbot
user_input = "Hello, how are you?"
response = handle_user_input(user_input)
print(response)
