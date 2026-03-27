import os
import json
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from tools import (
    search_products, get_inventory_health, get_pricing_analysis,
    get_review_insights, get_category_performance, generate_restock_alert
)

load_dotenv()

class Intent(str, Enum):
    INVENTORY = "INVENTORY"
    PRICING = "PRICING"
    REVIEWS = "REVIEWS"
    CATALOG = "CATALOG"
    GENERAL = "GENERAL"

class RouterOutput(BaseModel):
    intent: Intent = Field(description="The classified intent of the user query.")

class RetailMindAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY", "dummy"),
            temperature=0
        )
        
        # Tools grouped by intent
        self.inventory_tools = [get_inventory_health, generate_restock_alert]
        self.pricing_tools = [get_pricing_analysis]
        self.reviews_tools = [get_review_insights]
        self.catalog_tools = [search_products, get_category_performance]
        
        # Setup specific agents
        self.inventory_agent = self._create_agent(self.inventory_tools, "You are an Inventory Specialist for StyleCraft. Answer queries regarding stock, stockouts, and alerts.")
        self.pricing_agent = self._create_agent(self.pricing_tools, "You are a Pricing Analyst for StyleCraft. Answer queries about margins, pricing, and profitability.")
        self.review_agent = self._create_agent(self.reviews_tools, "You are a Customer Insight Analyst. Answer queries about reviews and sentiment for StyleCraft products.")
        self.catalog_agent = self._create_agent(self.catalog_tools, "You are a Catalog Manager for StyleCraft. Answer queries about product search and category rollups.")
        
        # Router setup
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert intent classifier. Route the user's query into one of these categories:\n"
             "INVENTORY: Questions about stock levels, stockout risk, restock needs.\n"
             "PRICING: Questions about margins, profitability, pricing tiers.\n"
             "REVIEWS: Questions about customer feedback, ratings, complaints.\n"
             "CATALOG: Questions about product search, top performers, category overviews.\n"
             "GENERAL: Greetings, retail knowledge, or follow-ups not needing tools."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        
        self.router = self.llm.with_structured_output(RouterOutput)
        
    def _create_agent(self, tools, system_instruction):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        
    def invoke(self, query: str, chat_history: list = None):
        if chat_history is None: chat_history = []
        
        # Route
        try:
            router_chain = self.router_prompt | self.router
            route_result = router_chain.invoke({"query": query, "chat_history": chat_history})
            intent = route_result.intent
        except Exception as e:
            # Fallback in case of structured output failure
            intent = Intent.GENERAL
            
        print(f"--- ROUTED TO: {intent.value} ---")
            
        # Dispatch
        try:
            if intent == Intent.INVENTORY:
                response = self.inventory_agent.invoke({"input": query, "chat_history": chat_history})["output"]
            elif intent == Intent.PRICING:
                response = self.pricing_agent.invoke({"input": query, "chat_history": chat_history})["output"]
            elif intent == Intent.REVIEWS:
                response = self.review_agent.invoke({"input": query, "chat_history": chat_history})["output"]
            elif intent == Intent.CATALOG:
                response = self.catalog_agent.invoke({"input": query, "chat_history": chat_history})["output"]
            else:
                # General conversation response using raw LLM
                messages = chat_history + [HumanMessage(content=query)]
                response = self.llm.invoke(messages).content
        except Exception as e:
            return f"Agent error occurred: {str(e)}", intent.value

        return response, intent.value
