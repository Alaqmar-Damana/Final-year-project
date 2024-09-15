from langchain_openai import ChatOpenAI
import os
# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI instance
chat = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo")

# Define the prompt for the GPT model
prompt = "Hello, how are you today?"

# Call the GPT model and get the response
response = chat.invoke(prompt)

# Print the generated text
print(response)