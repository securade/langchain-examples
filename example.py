from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template("You said: {user_input}. Here's a response:")

# Define the user input
user_input = "Hello, how are you?"

# Create the chat prompt
chat_prompt = prompt_template.format_prompt(user_input=user_input).to_messages()

# Generate the response using ChatOpenAI
response = ChatOpenAI().invoke(chat_prompt)

# Print the response
print(response.content)