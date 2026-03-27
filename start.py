import streamlit as st
import pandas as pd
from tools import generate_restock_alert, get_review_insights, products_df, category_means
from agent import RetailMindAgent
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="RetailMind Product Agent", layout="wide")

# Persistent Initialization
if 'agent' not in st.session_state:
    st.session_state.agent = RetailMindAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'briefing_generated' not in st.session_state:
    st.session_state.briefing_generated = False

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.briefing_generated = False
    
def generate_daily_briefing():
    st.markdown("### 📰 Daily RetailMind Briefing")
    
    # 1. Top 3 restock alerts
    alerts = generate_restock_alert.invoke({"threshold_days": 7})
    if alerts:
        st.error("**⚠️ Critical Inventory Alerts:**")
        for i, a in enumerate(alerts[:3]):
            st.write(f"{i+1}. **{a['product_name']}** (ID: {a['product_id']}) - Out in {a['days_to_stockout']} days. Rev at Risk: â¹{a['revenue_at_risk']}")
    else:
        st.success("âï¸ No critical stockouts in the next 7 days.")
        
    # 2. Worst rated product
    if not products_df.empty:
        worst = products_df.loc[products_df['avg_rating'].idxmin()]
        insights = get_review_insights.invoke({"product_id": worst['product_id']})
        st.warning(f"**📉 Lowest Rated Product:** {worst['product_name']} ({worst['avg_rating']}/5.0)")
        st.write(f" *Customer Feedback:* {insights['sentiment_summary']}")
        
    # 3. Margin Flag
    if not products_df.empty:
        products_df['margin_pct'] = ((products_df['price'] - products_df['cost']) / products_df['price']) * 100
        lowest_margin = products_df.loc[products_df['margin_pct'].idxmin()]
        if lowest_margin['margin_pct'] < 25:
            st.info(f"**💰 Margin Alert:** **{lowest_margin['product_name']}** has a gross margin of {lowest_margin['margin_pct']:.1f}%.")
            st.write(" *Suggested Action:* Review material sourcing costs or implement a strategic price increase to reach the 25% target.")
    st.divider()

# Sidebar
st.sidebar.title("Configuration")
category_filter = st.sidebar.selectbox("Scope Categories:", ["All Categories", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"])
if st.sidebar.button("Clear Chat", on_click=clear_chat):
    pass

# Catalog Summary Panel
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Catalog Summary")
if not products_df.empty:
    df = products_df
    if category_filter != "All Categories":
         df = products_df[products_df['category'].str.lower() == category_filter.lower()]
    
    total_skus = len(df)
    
    if total_skus > 0:
        df_margin = ((df['price'] - df['cost']) / df['price']) * 100
        avg_margin = df_margin.mean()
        avg_rating = df['avg_rating'].mean()
        df_days_to_stockout = df['stock_quantity'] / df['avg_daily_sales'].replace(0, 0.001)
        critical_count = len(df[df_days_to_stockout < 7])
        
        st.sidebar.metric("Total SKUs", total_skus)
        st.sidebar.metric("Critical Stock Items", critical_count)
        st.sidebar.metric("Avg Margin %", f"{avg_margin:.1f}%")
        st.sidebar.metric("Avg Rating", f"{avg_rating:.2f} ⭐")
    else:
        st.sidebar.write("No data available for this category.")

# Main Chat Interface
st.title("StyleCraft AI Agent")

# Render Briefing on start
if not st.session_state.briefing_generated:
    generate_daily_briefing()
    st.session_state.briefing_generated = True

# Display Chat History
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Input
if prompt := st.chat_input("Ask about your catalog, inventory, pricing, or reviews..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # If category filter is active, inject it as context
            context_prompt = prompt
            if category_filter != "All Categories":
                context_prompt = f"[Context: Focusing on {category_filter} category] " + prompt
                
            response, intent = st.session_state.agent.invoke(context_prompt, st.session_state.chat_history)
            st.write(response)
            if intent:
                st.caption(f"Routed via: {intent} Handler")
            st.session_state.chat_history.append(AIMessage(content=response))
