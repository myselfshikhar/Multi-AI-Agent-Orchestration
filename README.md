# 5-Agent Financial AI Assistant

An advanced, multi-agent AI assistant designed to answer complex financial questions using a modular 5-agent workflow. This system leverages LangChain and Google's Gemini Pro to perform data research, calculation, summarization, and critical analysis, providing users with comprehensive and validated financial insights.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Framework-yellow.svg)
![Google Gemini](https://img.shields.io/badge/Google-Gemini%20Pro-green.svg)

---

## 🚀 About The Project

This project simulates a team of specialist AI agents to provide nuanced and data-driven answers to financial queries. It moves beyond a single-prompt LLM by breaking down complex questions into specialized tasks, ensuring higher accuracy and reliability. The system can fetch financial data, perform calculations, summarize key points, identify risks, and synthesize all information into a single, cohesive answer.

A key feature is its **explainability logging**, which records the reasoning of each agent for every query, providing transparency into the AI's decision-making process.

### Core Features

* **Modular 5-Agent Architecture:** Implements a workflow with distinct Research, Summarizer, Calculator, Critic, and Synthesizer agents.
* **Context-Aware Memory:** Remembers the conversation history to accurately answer follow-up questions.
* **Enhanced Financial Tooling:** The Calculator agent can call dedicated Python functions for specific financial metrics like ROI.
* **Built-in Validation:** A Critic agent reviews all information to identify risks and inconsistencies.
* **Persistent Explainability Logs:** Automatically saves the output of each agent to a structured `explainability_logs.jsonl` file for every query.

---

## 🏛️ System Architecture

The assistant orchestrates a five-step workflow, where the output of one agent becomes the input for the next, ensuring a logical flow of analysis.


User Query
│
└───> 1. Research Agent (Fetches Data)
│
└───> 2. Summarizer Agent (Simplifies Data)
│
└───> 3. Calculator Agent (Performs Math)
│
└───> 4. Critic Agent (Validates & Finds Risks)
│
└───> 5. Synthesizer Agent (Creates Final Answer)
│
└───> Final Answer to User


---

## 🛠️ Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.9+
* A Google AI API Key, which you can obtain from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation & Setup

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/your-username/financial-agent-assistant.git](https://github.com/your-username/financial-agent-assistant.git)
    cd financial-agent-assistant
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    * **macOS/Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies**
    ```sh
    pip install langchain langchain-google-genai python-dotenv
    ```

4.  **Configure Your API Key**
    * Create a file named `.env` in the root of the project directory.
    * Add your Google AI API key to the `.env` file as follows:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```

---

## ▶️ Usage

With your environment configured, run the assistant from the project's root directory:

```sh
python main.py
```

The application will start, and you can begin asking financial questions.

Example Session
Your Question:

I'm thinking about investing in Apple. If I bought a share at $150 and it's now worth $175.50, what is my ROI? Based on its current data, what are the key takeaways and risks?
Terminal Output:
The program will first display the detailed logs from each agent, followed by the final synthesized answer.

--- Explainability Logs ---
**Calculator Agent**: Performed calculation: Return on Investment (ROI) is 17.00%.
**Critic Agent**: Provided critique: While the 17.00% ROI is strong, the positive analyst ratings and sales forecasts must be weighed against the noted risks of high competition and regulatory scrutiny.
**Research Agent**: Identified ticker 'AAPL' and fetched data: {'ticker': 'AAPL', 'price': 175.5, ...}
**Summarizer Agent**: Created summary: - Apple (AAPL) is currently trading at $175.50 with a Strong Buy rating from analysts...

--- Final Answer ---
Based on your numbers, the Return on Investment (ROI) for your Apple share would be 17.00%.

According to current data, Apple (AAPL) is viewed favorably by analysts with a "Strong Buy" rating, supported by positive news sentiment around future sales. However, it's important to consider the associated risks, which include intense competition within the smartphone market and ongoing regulatory scrutiny.
💡 Future Improvements
Integrate with real financial APIs (e.g., Yahoo Finance, Alpha Vantage) instead of the mock API.

Expand the Calculator agent's tools with more financial metrics (CAGR, Sharpe Ratio, etc.).

Develop a simple web interface using Flask or Streamlit.

Allow the Research agent to browse the web for news articles.




