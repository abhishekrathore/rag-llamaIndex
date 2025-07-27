import pandas as pd
from openai import OpenAI


## 1. Read data from Google Sheets
def read_data_frame(document_id, sheet_name):
    export_link = f"https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(export_link)


document_id = "14fKHsblfqZfWj3iAaM2oA51TlYfQlFT4WKo52fVaQ9U"
products_df = read_data_frame(document_id, "products")
emails_df = read_data_frame(document_id, "emails")

# Display first 3 rows of each DataFrame
print(products_df.head(3))
print(emails_df.head(3))
test_emails_df = emails_df.iloc[[6, 12]]
# TODO : testing with only top few email
print(test_emails_df)

## 2. Google Auth and creating spreadsheet


import gspread
from gspread_dataframe import set_with_dataframe

gc = gspread.oauth(
    credentials_filename="./client_secret_343193111877-rabmoeg6lrjb7eq6n25eivdjvqqmles2.apps.googleusercontent.com.json",
)


# This code goes after creating google client
output_document = gc.create("Solving Business Problems with AI - Output")

# Create 'email-classification' sheet
email_classification_sheet = output_document.add_worksheet(
    title="email-classification", rows=50, cols=2
)
email_classification_sheet.update([["email ID", "category"]], "A1:B1")


## 3. Open AI Setup
# Set your OpenAI API key
import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()
client = OpenAI()


## 4. Classification of Emails and adding to spreadsheet
def batch_classify_emails(subjects, messages, batch_size=5):
    results = []
    for i in range(0, len(subjects), batch_size):
        batch_subjects = subjects[i : i + batch_size]
        batch_messages = messages[i : i + batch_size]

        prompt = "Classify the following emails as either 'product inquiry' or 'order request'. Return only the categories in order:\n\n"
        for idx, (subj, msg) in enumerate(zip(batch_subjects, batch_messages), start=1):
            prompt += f"Email {idx}:\nSubject: {subj}\nMessage: {msg}\n\n"
        prompt += "Categories:"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        categories_text = response.choices[0].message.content.strip()

        # Parse categories: assume response is something like "1. product inquiry\n2. order request..."
        categories = [
            line.split(".")[-1].strip().lower()
            for line in categories_text.splitlines()
            if "." in line
        ]
        if len(categories) < len(batch_subjects):
            # Fallback: simple line-by-line
            categories = [line.strip().lower() for line in categories_text.splitlines()]
        results.extend(categories)

    return results


test_emails_df["category"] = batch_classify_emails(
    test_emails_df["subject"].tolist(), test_emails_df["message"].tolist()
)

## Add Dataframe to Email-classification sheet (rename columns as per the output requirement)
final_result = test_emails_df[["email_id", "category"]]
final_result = final_result.rename(
    columns={"email_id": "email ID", "category": "Category"}
)
set_with_dataframe(email_classification_sheet, final_result)

print(f"Shareable link: https://docs.google.com/spreadsheets/d/{output_document.id}")


## 5. Create document data for LlamaIndex = emailIndex
email_texts = []
email_metadata_list = []

for _, row in test_emails_df.iterrows():
    text = f"Email Id: {row['email_id']}\nSubject: {row['subject']}\nMessage: {row['message']}\nCategory: {row['category']}"
    email_texts.append(text)
    email_metadata_list.append(
        {"email_id": row["email_id"], "category": row["category"]}
    )


## 6. Create document data for LlamaIndex = productIndex

product_texts = []
product_metadata_list = []
for _, row in products_df.iterrows():
    text = f"name: {row['name']}\ncategory: {row['category']}\ndescription: {row['description']}"
    product_texts.append(text)
    metadata = {
        "product_id": row["product_id"],
        "name": row["name"],
        "category": row["category"],
        "seasons": row["seasons"],
    }
    product_metadata_list.append(metadata)


## 7. Create LlamaIndex for emails and products

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    VectorStoreIndex,
    Document,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding

# Setting for Embedding and LLM model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o")

email_docs = [
    Document(text=t, metadata=m) for t, m in zip(email_texts, email_metadata_list)
]

product_docs = [
    Document(text=t, metadata=m, id=m["product_id"])
    for t, m in zip(product_texts, product_metadata_list)
]

if not os.path.exists("storage"):
    # email index
    emailIndex = VectorStoreIndex.from_documents(email_docs)
    emailIndex.set_index_id("email_index")
    emailIndex.storage_context.persist("./storage")
    # product index
    productIndex = VectorStoreIndex.from_documents(product_docs)
    productIndex.set_index_id("product_index")
    productIndex.storage_context.persist("./storage")
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    # load index
    emailIndex = load_index_from_storage(storage_context, index_id="email_index")
    productIndex = load_index_from_storage(storage_context, index_id="product_index")

## 8. Query Engine with filters
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="order request"),
    ],
)

email_query_engine = emailIndex.as_query_engine(filters=filters, similarity_top_k=5)

## 9. Prompt for Getting Product ID and quantity from an Email

query_prompt = """**Task**: For each document:
1. Product ID (format: 3 uppercase letters + 4 digits, e.g., ABC1234)
2. Quantity requested (numeric value)

**Instructions**:
- For each document:
  a. Identify product IDs matching the pattern [A-Z]{3}\\d{4}
  b. Extract quantities (convert written numbers to digits)
  c. If quantity unspecified, make it 1
  d. only return single number in quantity, choose higher one
  e. There can be multiple product IDs in text
**Output Format**:
For each document and for each Product ID in it, return EXACTLY :
Email ID: [ID]
Product ID: [ID or "NOT_FOUND"]
Quantity: [number]

example :
Email ID: E003 | Product ID: SFT1098 | Quantity: 3

Note: There can be multiple output lines for document which have more than 1 product IDs
"""

response = email_query_engine.query(query_prompt)
print("LLM response for Product IDs :\n")
print(response)


## 10. Convert Response to Order Dataframe
import re

pattern = r"Email ID:\s*(?P<Email_ID>\w+)\s*\|\s*Product ID:\s*(?P<Product_ID>\w+)\s*\|\s*Quantity:\s*(?P<Quantity>\d+)"

data = []

lines = response.response.splitlines()
for text in lines:
    match = re.search(pattern, text)
    if match:
        data.append(
            {
                "Email_ID": match.group("Email_ID"),
                "Product_ID": match.group("Product_ID"),
                "Quantity": int(match.group("Quantity")),
            }
        )

order_df = pd.DataFrame(data)
print("ORDER dataframe :\n")
print(order_df)


## 11. SQL Lite database for Products

import sqlite3

database = "products.db"

conn = sqlite3.connect(database)
# Insert DataFrame to SQLite
products_df.to_sql("products", conn, if_exists="replace", index=False)

print("\nData inserted successfully!")

# Verify the data was inserted correctly
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("\nTables in database:", cursor.fetchall())

cursor.execute("SELECT COUNT(*) FROM products;")
print("\nNumber of rows in products table:", cursor.fetchone()[0])

conn.close()


## 12. DB SQL alchemy setup and Query functions

from operator import and_
from sqlalchemy import create_engine, select, text, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import String, Integer, Float, Text
from typing import Optional
from sqlalchemy.exc import SQLAlchemyError


# Base class for declarative models
class Base(DeclarativeBase):
    pass


# Product model
class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    category: Mapped[Optional[str]] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(Text)
    stock: Mapped[Optional[int]] = mapped_column(Integer)
    seasons: Mapped[Optional[str]] = mapped_column(String)
    price: Mapped[Optional[float]] = mapped_column(Float)

    def __repr__(self):
        return f"Product(id={self.product_id!r}, name={self.name!r}, price={self.price}, stock={self.stock})"


# Database setup
engine = create_engine("sqlite:///products.db")
session = Session(engine)


# Query functions
def get_product_by_id(product_id: str) -> Optional[Product]:
    stmt = select(Product).where(Product.product_id == product_id)
    return session.scalar(stmt)


def get_product_by_id_with_stock(product_id: str, quantity: int) -> Optional[Product]:
    stmt = select(Product).where(
        and_(Product.product_id == product_id, Product.stock >= quantity)
    )
    return session.scalar(stmt)


def get_product_dict(product_id: str) -> dict | None:
    product = session.get(Product, product_id)
    if product:
        return {
            column.name: getattr(product, column.name)
            for column in Product.__table__.columns
        }
    return None


def update_product_stock(product_id: str, quantity: int) -> bool:
    try:
        result = session.execute(
            update(Product)
            .where(and_(Product.product_id == product_id, Product.stock >= quantity))
            .values(stock=Product.stock - quantity)
        )
        print(quantity)
        return result.rowcount == 1

    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return False


## 13. Checking if Stock is available for 'order request'

status = []
# print(get_product_by_id('SFT1098'))
for index, row in order_df.iterrows():
    email_id = row["Email_ID"]
    product_id = row["Product_ID"]
    quantity = row["Quantity"]
    current_product = get_product_by_id(product_id)
    print(f"Initial Stock {product_id}:", current_product)

    res = update_product_stock(product_id, quantity)
    if res:
        print(f"\nThis order item is fulfilled", email_id, product_id, quantity)
        status.append("created")
    else:
        print(
            f"\nThis order item is out of stock ",
            email_id,
            product_id,
            f"item reqd quantity :  {quantity} but only {current_product['stock']} is/are available",
        )
        status.append("out of stock")
    session.commit()


for index, row in order_df.iterrows():
    product_id = row["Product_ID"]
    print(f"Final Stock {product_id}:", get_product_by_id(product_id))


order_df["status"] = status
order_df = order_df.rename(
    columns={"Email_ID": "email ID", "Product_ID": "product Id", "Quantity": "quantity"}
)

print(order_df)


## 14. Query for getting similar product recommendation


def get_similar_products(
    target_product_id: str, target_product_name: str, retriever
) -> list:

    # Query using the product name as the search text
    # TODO : we can improve similarity search here
    retrieved_nodes = retriever.retrieve(target_product_name)

    similar_products = []
    for node in retrieved_nodes:
        product_data = node.metadata
        if product_data["product_id"] != target_product_id:
            similar_products.append({"product": product_data, "score": node.score})

    return similar_products[:5]  # Return top 5 similar products


## 15. Group order items by email ID - so that complete order info is available in Single Email response


grouped_by_email = order_df.groupby('email ID')

all_orders = []
for email, group in grouped_by_email:
    print(f"\nEmail: {email}")
    email_info = emails_df[emails_df['email_id']==email].to_dict('records')[0]
    print(email_info)
    complete_order_info = {'email':email, 'products':[]}
    for index, row in group.iterrows():
        product = get_product_dict({row['product Id']})
        status = row['status']
        quantity = row['quanitity']
        complete_order_info['products'].append({'info':product,'status':status,'quantity':quantity})
    all_orders.append(complete_order_info)

print(all_orders)


 ## 16. Finding alternative for a product based on similarity and stock availability
 
product_ret = productIndex.as_retriever(similarity_top_k=3)

def get_alternatives(product_id, product_name, product_cat, quantity, retriever):
# Similar product recommendations
    recommendations = []
    similar_items = get_similar_products(
        product_id, product_name, product_cat,retriever
    )

    print(
        f"Products similar to '{product_name}' having quanity more than or equal to {quantity}:"
    )

    for idx, item in enumerate(similar_items, 1):
        p = item["product"]

        # CHECK QUANTITY IS IN STOCK in SQL

        if get_product_by_id_with_stock(p["product_id"], quantity):
            recommendations.append(p)
    return recommendations

print(get_alternatives('CLF2109','Cable Knit Beanie','Accessories',2,product_ret))

# final_order_status_df.query('status == "created"')
email_responses = {'email ID':[],'response':[]}


for order in all_orders:
    products = order['products']
    email = order['email']
    # created_count = sum(1 for product in products if product["status"] == "created")
    # print(created_count)
    # out_of_stock_count = sum(1 for product in products if product["status"] == "out of stock")
    # print(out_of_stock_count)

    out_of_stock = [product for product in products if product["status"] == "out of stock"]
    for product in out_of_stock:
        product['info']['alternative'] = get_alternatives(product['info']["product_id"], product['info']["product_name"], product['info']["category"],product['quantity'],product_ret)


    order_success_prompt = f"""
        You are an email assistant for an online retail store. Your task is to generate a personalized order confirmation email based on the customer’s order data, which will be provided in JSON format.

        ---

        JSON Input Format:
        {
          "customer_message": {
            "subject": email['subject'],
            "message": email['message']
          },
          "shipped_items": [product.info for product in products if product["status"] == "created"],
          "out_of_stock_items": [
            {
              "product_name": "Cable Knit Beanie",
              "product_id": "CLF2109",
              "requested_quantity": 5,
              "alternative": {
                "product_id": "VSC6789",
                "product_name": "Versatile Scarf"
              }
            },
            {
              "product_name": "Winter Scented Candle",
              "product_id": "SNC3456",
              "requested_quantity": 3,
              "alternative": {
                "product_id": "HSC7890",
                "product_name": "Holiday Spice Candle"
              }
            }
          ]
        }

        ---

        Your task is to use the JSON input to generate a personalized email that:

        1. Greets the customer using their name if available (from the message); otherwise, use "Dear Customer."
        2. Confirms whether the order is fully shipped, partially shipped, or completely out of stock, depending on the number of items shipped.
        3. Lists each shipped item with:
          - Product name
          - Product ID
          - Price per unit
          - Quantity
          - Subtotal (price * quantity)
        4. Mentions each out-of-stock item and recommends the alternative item (from the "alternative" field) that can still be shipped.
        5. Includes a short, friendly message inspired by the customer's original note (e.g., holiday gifts, comfort items).
        6. Calculates and displays the total cost of the shipped items.
        7. Uses a warm, professional, and enthusiastic tone.
        8. Keeps the email concise and ends with a thank-you. \n"""

        print(order_success_prompt)
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": order_success_prompt}],
    #     temperature=0
    # )
    
   

print(email_responses)   
