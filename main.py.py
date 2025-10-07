# # ==============================================================================
# # 1. IMPORTS AND SETUP
# # ==============================================================================
# # We import necessary libraries.
# # 'os' is for interacting with the operating system, like getting environment variables.
# # 'dotenv' is for loading those environment variables from a .env file.
# import os
# from dotenv import load_dotenv

# # LangChain libraries provide the building blocks for our AI application.
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser

# # --- Load Environment Variables ---
# # This line loads the GOOGLE_API_KEY from your .env file so the script can use it.
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# # A quick check to make sure the API key was actually found.
# if not api_key:
#     raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in your .env file.")


# # ==============================================================================
# # 2. INITIALIZE THE LARGE LANGUAGE MODEL (LLM) AND MEMORY
# # ==============================================================================

# # --- Initialize the LLM ---
# # This creates the connection to Google's Gemini model.
# # We use "gemini-1.5-flash-latest" which is fast and efficient.
# # 'temperature=0.7' makes the model's responses creative but not too random.
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.7)

# # --- Initialize Memory ---
# # This creates a memory object for our conversation.
# # It will store the history of questions and answers, allowing the AI to have context.
# # 'memory_key="history"' gives a name to the conversation history variable.
# memory = ConversationBufferMemory(return_messages=True, memory_key="history")


# # ==============================================================================
# # 3. DEFINE THE AGENT CHAINS
# # ==============================================================================
# # Instead of separate functions, we now define "chains" using LangChain's
# # modern Expression Language (LCEL). A chain pipes the output of one
# # component into the next (e.g., prompt -> llm -> output).

# # --- Calculator Agent ---
# # This agent is a specialist for mathematical or numerical questions.
# calculator_prompt = PromptTemplate(
#     template="As a finance calculator, answer: {query}. Return only the calculation or final numerical value.",
#     input_variables=["query"]
# )
# calculator_chain = calculator_prompt | llm | StrOutputParser()

# # --- Knowledge Agent (Now with Memory!) ---
# # This is our main expert. It now takes both the conversation history and the
# # new query as input, making it context-aware for follow-up questions.
# knowledge_prompt = PromptTemplate(
#     template="""As a finance expert, consider the previous conversation context below to answer the user's question. If the context is not relevant, ignore it.

# <Conversation_History>
# {history}
# </Conversation_History>

# User's new question: {query}

# Provide a concise answer:""",
#     input_variables=["history", "query"]
# )
# knowledge_chain = knowledge_prompt | llm | StrOutputParser()

# # --- Aggregator Agent ---
# # This agent's job is to take the outputs from all other agents and synthesize
# # them into a single, clean, final answer for the user.
# aggregator_prompt = PromptTemplate(
#     template="""You have received outputs from two specialist agents. Your task is to combine them into a single, helpful, and concise final answer.

# Calculator Agent Output:
# {calc_output}

# Knowledge Agent Output:
# {knowledge_output}

# Based on these, provide the best possible final answer to the user. Do not say "The calculator agent said...". Just give the direct, final answer.""",
#     input_variables=["calc_output", "knowledge_output"]
# )
# aggregator_chain = aggregator_prompt | llm | StrOutputParser()


# # ==============================================================================
# # 4. THE CORE ORCHESTRATOR FUNCTION
# # ==============================================================================
# # This function manages the entire process for a single query.
# def process_finance_query(query: str):
#     """
#     Orchestrates the query process:
#     1. Loads past conversation history.
#     2. Runs the calculator and knowledge agents.
#     3. Aggregates their outputs into a final answer.
#     4. Saves the new question and answer to memory.
#     """
#     # Step 1: Load the conversation history from memory.
#     # We get the history and convert it to a simple string.
#     chat_history_dict = memory.load_memory_variables({})
#     chat_history_str = str(chat_history_dict.get('history', 'This is the beginning of the conversation.'))

#     # Step 2: Run the specialist agents in parallel.
#     # The calculator handles any math in the query.
#     calc_output = calculator_chain.invoke({"query": query})
#     # The knowledge agent gets both history and the new query for context.
#     knowledge_output = knowledge_chain.invoke({"history": chat_history_str, "query": query})

#     # Step 3: Run the aggregator agent to get the final answer.
#     # It takes the outputs from the other agents to create the final response.
#     final_answer = aggregator_chain.invoke({
#         "calc_output": calc_output,
#         "knowledge_output": knowledge_output
#     })

#     # Step 4: Save the current interaction to memory for the next turn.
#     # This ensures the AI will remember this question and answer in the future.
#     memory.save_context({"input": query}, {"output": final_answer})

#     # Step 5: Return only the final answer.
#     return final_answer


# # ==============================================================================
# # 5. MAIN EXECUTION LOOP
# # ==============================================================================
# # This block runs when you execute the script directly (e.g., "python demo.py").
# if __name__ == "__main__":
#     print("Welcome to your Financial AI Assistant (powered by Google Gemini).")
#     print("Type 'exit' to end the conversation.")

#     # This loop will run continuously, waiting for user input.
#     while True:
#         # Ask the user for their question.
#         user_query = input("\nYour question: ").strip()

#         # If the user types 'exit', break the loop and end the program.
#         if user_query.lower() == 'exit':
#             print("Goodbye!")
#             break

#         # If there's a query, process it.
#         if user_query:
#             try:
#                 # Call the main orchestrator function to get the AI's response.
#                 final_result = process_finance_query(user_query)

#                 # --- CLEAN OUTPUT ---
#                 # Only the final, polished answer is shown to the user.
#                 print(f"\nAssistant: {final_result}")

#             except Exception as e:
#                 # In case of an error (e.g., API issue), print a helpful message.
#                 print(f"\nAn error occurred: {e}")
#                 print("Please check your API key and network connection.")


#                 # Explain what compound interest is.
#                 # Okay, so if I invest $5,000 at an average annual return of 7%, how much would it be worth in 20 years?
#                 # That's great. And how much of that total amount is just the profit from interest?
#                 #What is diversification, and can you suggest a simple portfolio allocation for that $5,000 for a moderate-risk investor?






# main.py

# ==============================================================================
# 1. IMPORTS AND SETUP
# ==============================================================================
import os
import json # Used for creating structured explainability logs
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in your .env file.")

# ==============================================================================
# 2. INITIALIZE LLM AND MEMORY
# ==============================================================================
# The core language model that will power all our agents
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.7)
# The conversational memory, which stores the final input/output of each turn
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# ==============================================================================
# 3. MOCK APIs & FINANCIAL TOOLS (As per Mentor Feedback)
# ==============================================================================
# --- Mock Research API ---
# In a real-world scenario, this would connect to Yahoo Finance, Alpha Vantage, etc.
# For this demo, it returns pre-defined data for the ticker 'AAPL'.
def mock_research_api(ticker: str) -> dict:
    """Simulates fetching financial data for a given stock ticker."""
    print(f"--- [External Tool] Mock Research API called for ticker: {ticker} ---")
    if ticker.upper() == "AAPL":
        return {
            "ticker": "AAPL",
            "price": 175.50,
            "pe_ratio": 29.5,
            "market_cap": "2.8T",
            "analyst_rating": "Strong Buy",
            "news_sentiment": "Positive: Strong iPhone 17 sales forecast.",
            "risks": "High competition in the smartphone market; regulatory scrutiny."
        }
    else:
        return {"error": f"Data for ticker '{ticker}' not found in mock API."}

# --- Dedicated Financial Calculator Tools ---
# These functions handle specific financial math, making the Calculator Agent more reliable.
def calculate_roi(initial_investment: float, final_value: float) -> float:
    """Calculates the Return on Investment (ROI)."""
    print("--- [External Tool] 'calculate_roi' function called ---")
    return ((final_value - initial_investment) / initial_investment) * 100

# ==============================================================================
# 4. THE 5 AGENT DEFINITIONS
# ==============================================================================
# Each agent is a chain with a specific prompt and purpose.

# --- Agent 1: Research Agent ---
# Identifies the key entity (e.g., stock ticker) in the query and fetches data.
research_prompt = PromptTemplate(
    template="Your task is to identify the company stock ticker from the user's query. Return ONLY the ticker symbol. For example, if the query is 'Tell me about Apple', you should return 'AAPL'. Query: {query}",
    input_variables=["query"]
)
research_chain = research_prompt | llm | StrOutputParser()

# --- Agent 2: Calculator Agent (Enhanced) ---
# Can now perform simple math or request a specific tool for complex financial calculations.
calculator_prompt = PromptTemplate(
    template="""You are a calculator. Analyze the user's query to determine if a specific financial calculation is needed.

If the query requires a Return on Investment (ROI) calculation, respond with a JSON object like this:
{{"tool": "roi", "initial": <initial_value>, "final": <final_value>}}

For any other simple math, just calculate and return the numerical answer.

Query: {query}""",
    input_variables=["query"]
)
calculator_chain = calculator_prompt | llm | StrOutputParser()

# --- Agent 3: Summarizer Agent ---
# Takes raw data (from the Research Agent) and makes it easy to understand.
summarizer_prompt = PromptTemplate(
    template="You are a financial analyst. Summarize the following raw financial data in 2-3 simple, human-readable bullet points. Data: {research_data}",
    input_variables=["research_data"]
)
summarizer_chain = summarizer_prompt | llm | StrOutputParser()

# --- Agent 4: Critic Agent ---
# Reviews the outputs of other agents to validate them and identify risks.
critic_prompt = PromptTemplate(
    template="""You are a financial critic. Review the research summary and the calculation result.
Identify any potential risks, inconsistencies, or important considerations that are missing.
Provide your critique in 1-2 sharp, insightful sentences.

Research Summary:
{summary}

Calculation Result:
{calculation}
""",
    input_variables=["summary", "calculation"]
)
critic_chain = critic_prompt | llm | StrOutputParser()

# --- Agent 5: Synthesizer Agent (Final Response) ---
# The final agent that assembles all information into a single, cohesive answer.
synthesizer_prompt = PromptTemplate(
    template="""You are a helpful financial assistant. Your goal is to provide a complete and balanced final answer to the user.
Combine the information from the Research, Calculator, and Critic agents into a final, helpful response.

User's Original Question: {query}

Research Summary:
{summary}

Calculation Result:
{calculation}

Expert Critique & Risks:
{critique}

Provide the final, synthesized answer below:
""",
    input_variables=["query", "summary", "calculation", "critique"]
)
synthesizer_chain = synthesizer_prompt | llm | StrOutputParser()

# ==============================================================================
# 5. THE CORE ORCHESTRATOR FUNCTION
# ==============================================================================
def process_finance_query(query: str):
    """
    Orchestrates the full 5-agent workflow and generates explainability logs.
    """
    # This dictionary will store the output of each agent for logging.
    explainability_logs = {}

    # --- Step 1: Research Agent ---
    ticker = research_chain.invoke({"query": query})
    research_data = mock_research_api(ticker)
    explainability_logs["Research Agent"] = f"Identified ticker '{ticker}' and fetched data: {research_data}"

    # --- Step 2: Summarizer Agent ---
    summary = summarizer_chain.invoke({"research_data": json.dumps(research_data)})
    explainability_logs["Summarizer Agent"] = f"Created summary: {summary}"

    # --- Step 3: Calculator Agent ---
    calc_request = calculator_chain.invoke({"query": query})
    calculation_result = "Not applicable for this query."
    try:
        # Check if the agent requested a tool call
        request_json = json.loads(calc_request)
        if request_json.get("tool") == "roi":
            roi = calculate_roi(request_json["initial"], request_json["final"])
            calculation_result = f"Return on Investment (ROI) is {roi:.2f}%."
    except (json.JSONDecodeError, KeyError):
        # If it's not a valid JSON tool call, treat it as a simple math answer
        calculation_result = calc_request
    explainability_logs["Calculator Agent"] = f"Performed calculation: {calculation_result}"

    # --- Step 4: Critic Agent ---
    critique = critic_chain.invoke({"summary": summary, "calculation": calculation_result})
    explainability_logs["Critic Agent"] = f"Provided critique: {critique}"

    # --- Step 5: Synthesizer Agent ---
    final_answer = synthesizer_chain.invoke({
        "query": query,
        "summary": summary,
        "calculation": calculation_result,
        "critique": critique
    })

    # Save the final, user-facing interaction to memory
    memory.save_context({"input": query}, {"output": final_answer})

    # --- NEW: Save logs to a JSON file for permanent storage ---
    # We create a dictionary to hold the query and its corresponding logs.
    log_entry = {
        "user_query": query,
        "agent_logs": explainability_logs
    }
    # We open 'explainability_logs.jsonl' in append mode ('a').
    # This adds the new log without erasing old ones. Using .jsonl is a common practice for streaming JSON objects.
    with open("explainability_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n") # Convert the dictionary to a JSON string and write it.

    return final_answer, explainability_logs

# ==============================================================================
# 6. MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == "__main__":
    print("Welcome to the 5-Agent Financial AI Assistant.")
    print("Type 'exit' to end the conversation.")

    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        if user_query:
            try:
                # Get both the final answer and the detailed logs
                final_result, logs = process_finance_query(user_query)

                # --- Display Explainability Logs (As per mentor feedback) ---
                print("\n--- Explainability Logs ---")
                # Sort the logs for a consistent, logical order
                for agent_name, log_entry in sorted(logs.items()):
                    print(f"**{agent_name}**: {log_entry}")

                # --- Display Final Answer (As per your request) ---
                print("\n--- Final Answer ---")
                print(final_result)

            except Exception as e:
                print(f"\nAn error occurred: {e}")