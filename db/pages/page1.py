# pages/1_Page_1.py
import streamlit as st
import pandas as pd
from mysql.connector import Error

st.set_page_config(
    page_title="Products Management",
    page_icon="📘",
    layout="wide"
)

st.title("Products Management")
st.write("View and manage products in the database.")


# Function to get all products
def get_products():
    try:
        if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
            conn = st.session_state['db_connection']
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products")
            result = cursor.fetchall()

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Create DataFrame
            df = pd.DataFrame(result, columns=columns)
            return df
        else:
            return None
    except Error as e:
        st.error(f"Error retrieving products: {e}")
        return None


# Function to add a new product
def add_product(name):
    try:
        if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
            conn = st.session_state['db_connection']
            cursor = conn.cursor()

            # Insert new product
            query = "INSERT INTO products (name) VALUES (%s)"
            cursor.execute(query, (name,))
            conn.commit()

            return True, "Product added successfully!"
        else:
            return False, "No database connection available"
    except Error as e:
        return False, f"Error adding product: {e}"


# Check database connection status
if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
    if st.session_state['db_connection'].is_connected():
        st.success("Connected to database")

        # Display current products
        st.subheader("Current Products")
        products_df = get_products()

        if products_df is not None and not products_df.empty:
            st.dataframe(products_df)
        elif products_df is not None and products_df.empty:
            st.info("No products found in the database.")
        else:
            st.warning("Failed to retrieve products.")

        # Add new product section
        st.subheader("Add New Product")
        with st.form("add_product_form"):
            product_name = st.text_input("Product Name")
            submit_button = st.form_submit_button("Add Product")

            if submit_button and product_name:
                success, message = add_product(product_name)
                if success:
                    st.success(message)
                    # Refresh the products list by rerunning the app
                    st.experimental_rerun()
                else:
                    st.error(message)
            elif submit_button and not product_name:
                st.warning("Please enter a product name.")
    else:
        st.error("Database connection was established but is no longer active")
else:
    st.warning("No database connection available. Return to main page to connect.")

st.markdown("---")
st.write("Navigate using the sidebar to move between pages.")