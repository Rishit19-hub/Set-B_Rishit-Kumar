import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Load Data
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    products_df = pd.read_csv(os.path.join(current_dir, "retailmind_products.csv"))
    reviews_df = pd.read_csv(os.path.join(current_dir, "retailmind_reviews.csv"))
except Exception as e:
    print(f"Error loading CSV files: {e}")
    products_df, reviews_df = pd.DataFrame(), pd.DataFrame()

# Helper dict for easy category mean prices
category_means = products_df.groupby('category')['price'].mean().to_dict() if not products_df.empty else {}

# Initialize LLM for the review insights tool
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY", "dummy"),
    temperature=0
)

@tool
def search_products(query: str, category: str = None) -> list[dict]:
    """Searches and returns matching products from the CSV based on a text query and optional category filter. Returns top 5 matches."""
    if products_df.empty: return []
    df = products_df.copy()
    if category and category.lower() != "all":
        df = df[df['category'].str.lower() == category.lower()]
    
    # Simple substring search over multiple useful columns
    query_lower = query.lower()
    mask = (df['product_name'].str.lower().str.contains(query_lower)) | \
           (df['product_id'].str.lower().str.contains(query_lower)) | \
           (df['category'].str.lower().str.contains(query_lower))
    
    matches = df[mask].head(5)
    results = []
    for _, row in matches.iterrows():
        results.append({
            "product_id": row['product_id'],
            "product_name": row['product_name'],
            "category": row['category'],
            "price": row['price'],
            "stock_quantity": int(row['stock_quantity']),
            "avg_rating": row['avg_rating']
        })
    return results

@tool
def get_inventory_health(product_id: str) -> dict:
    """Returns inventory status for a product: current stock, average daily sales, estimated days to stockout, and a status flag (Critical <7, Low 7-14, Healthy >14)."""
    if products_df.empty: return {"error": "Product data unavailable"}
    product = products_df[products_df['product_id'].str.upper() == product_id.upper()]
    if product.empty: return {"error": f"Product with ID {product_id} not found."}
    
    row = product.iloc[0]
    stock = int(row['stock_quantity'])
    sales = float(row['avg_daily_sales'])
    
    days_to_stockout = stock / sales if sales > 0 else 999
    
    if days_to_stockout < 7:
        status = "Critical"
    elif days_to_stockout <= 14:
        status = "Low"
    else:
        status = "Healthy"
        
    return {
        "product_id": row['product_id'],
        "product_name": row['product_name'],
        "current_stock": stock,
        "avg_daily_sales": sales,
        "days_to_stockout": round(days_to_stockout, 1),
        "status_flag": status
    }

@tool
def get_pricing_analysis(product_id: str) -> dict:
    """Returns pricing intelligence: gross margin %, price positioning (Premium/Mid-Range/Budget), and a flag if margin is < 20%."""
    if products_df.empty: return {"error": "Product data unavailable"}
    product = products_df[products_df['product_id'].str.upper() == product_id.upper()]
    if product.empty: return {"error": f"Product with ID {product_id} not found."}
    
    row = product.iloc[0]
    price = float(row['price'])
    cost = float(row['cost'])
    
    gross_margin = ((price - cost) / price) * 100 if price > 0 else 0
    margin_flag = True if gross_margin < 20 else False
    
    cat_avg = category_means.get(row['category'], price)
    
    if price > cat_avg * 1.2:
        positioning = "Premium"
    elif price < cat_avg * 0.8:
        positioning = "Budget"
    else:
        positioning = "Mid-Range"
        
    return {
        "product_id": row['product_id'],
        "product_name": row['product_name'],
        "gross_margin_percent": round(gross_margin, 2),
        "price_positioning": positioning,
        "low_margin_flag": margin_flag
    }

@tool
def get_review_insights(product_id: str) -> dict:
    """Uses an LLM to summarize customer reviews for a given product. Returns avg rating, total reviews, sentiment summary, and top themes."""
    if reviews_df.empty or products_df.empty: return {"error": "Review data unavailable"}
    
    product = products_df[products_df['product_id'].str.upper() == product_id.upper()]
    if product.empty: return {"error": f"Product with ID {product_id} not found."}
    p_row = product.iloc[0]
    
    reviews = reviews_df[reviews_df['product_id'].str.upper() == product_id.upper()]
    if reviews.empty:
        return {
            "product_id": product_id,
            "product_name": p_row['product_name'],
            "avg_rating": p_row['avg_rating'],
            "total_reviews": 0,
            "sentiment_summary": "No reviews available.",
            "top_themes": []
        }
        
    reviews_text = ""
    for _, r in reviews.iterrows():
        reviews_text += f"- [{r['rating']}/5] {r['review_title']}: {r['review_text']}\n"
        
    prompt = f"""
    Analyze the following customer reviews for '{p_row['product_name']}'.
    Provide:
    1. A 2-sentence summary of the overall sentiment.
    2. The top 2 recurring themes (one positive, one negative if possible).
    Format the output strictly as:
    Summary: <text>
    Themes: <theme1>, <theme2>
    
    Reviews:
    {reviews_text}
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        lines = content.split('\n')
        summary = "Summary unavailable"
        themes = "Themes unavailable"
        for line in lines:
            if line.startswith("Summary:"): summary = line.replace("Summary:", "").strip()
            if line.startswith("Themes:"): themes = line.replace("Themes:", "").strip()
    except Exception as e:
        summary = f"Error during LLM generation: {e}"
        themes = "N/A"
        
    return {
        "product_id": product_id,
        "product_name": p_row['product_name'],
        "avg_rating": p_row['avg_rating'],
        "total_reviews": len(reviews),
        "sentiment_summary": summary,
        "top_themes": themes
    }

@tool
def get_category_performance(category: str) -> dict:
    """Returns aggregated category metrics: total SKUs, avg rating, avg margin %, total stock, critical stock count, top 3 revenue items."""
    if products_df.empty: return {"error": "Product data unavailable"}
    
    df = products_df[products_df['category'].str.lower() == category.lower()]
    if df.empty: return {"error": f"Category {category} not found."}
    
    total_skus = len(df)
    avg_rating = df['avg_rating'].mean()
    total_stock = df['stock_quantity'].sum()
    
    df['margin'] = (df['price'] - df['cost']) / df['price'] * 100
    avg_margin = df['margin'].mean()
    
    df['days_to_stockout'] = df['stock_quantity'] / df['avg_daily_sales'].replace(0, 0.001)
    critical_stock_count = len(df[df['days_to_stockout'] < 7])
    
    df['revenue_rate'] = df['price'] * df['avg_daily_sales']
    top_3 = df.nlargest(3, 'revenue_rate')[['product_name', 'revenue_rate']]
    top_3_list = [f"{row['product_name']} (Daily Rev: {row['revenue_rate']:.2f})" for _, row in top_3.iterrows()]
    
    return {
        "category": category,
        "total_skus": total_skus,
        "avg_rating": round(avg_rating, 2),
        "avg_margin_percent": round(avg_margin, 2),
        "total_stock": int(total_stock),
        "critical_stock_items": critical_stock_count,
        "top_3_revenue_products": top_3_list
    }

@tool
def generate_restock_alert(threshold_days: int = 7) -> list[dict]:
    """Scans all products and returns a list of products at risk of stockout within specified days, sorted by urgency."""
    if products_df.empty: return []
    
    alerts = []
    for _, row in products_df.iterrows():
        stock = int(row['stock_quantity'])
        sales = float(row['avg_daily_sales'])
        price = float(row['price'])
        
        days_to_stockout = stock / sales if sales > 0 else 999
        
        if days_to_stockout <= threshold_days:
            # Revenue at risk = price * (unmet demand during the threshold period)
            lost_units = (sales * threshold_days) - stock
            revenue_risk = price * lost_units if lost_units > 0 else 0
            
            alerts.append({
                "product_id": row['product_id'],
                "product_name": row['product_name'],
                "days_to_stockout": round(days_to_stockout, 1),
                "revenue_at_risk": round(revenue_risk, 2)
            })
            
    # Sort by urgency (fewest days remaining first)
    alerts = sorted(alerts, key=lambda x: x['days_to_stockout'])
    return alerts
