# pages/2_Page_2.py
import streamlit as st
import pandas as pd
from mysql.connector import Error

st.set_page_config(
    page_title="Order Form",
    page_icon="📝",
    layout="wide"
)

st.title("Order Form")
st.write("Create new orders by selecting products from the dropdown.")


# Function to get all products for dropdown
def get_products():
    try:
        if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
            conn = st.session_state['db_connection']
            cursor = conn.cursor()
            cursor.execute("SELECT id, name FROM products")
            result = cursor.fetchall()

            # Convert to dictionary for dropdown {name: id}
            products_dict = {row[1]: row[0] for row in result}
            return products_dict
        else:
            return None
    except Error as e:
        st.error(f"Error retrieving products: {e}")
        return None


# Function to add a new order
def add_order(product_id, quantity):
    try:
        if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
            conn = st.session_state['db_connection']
            cursor = conn.cursor()

            # Insert new order
            query = "INSERT INTO orders (product_id, quantity) VALUES (%s, %s)"
            cursor.execute(query, (product_id, quantity))
            conn.commit()

            return True, "Order added successfully!"
        else:
            return False, "No database connection available"
    except Error as e:
        return False, f"Error adding order: {e}"


# Function to get all orders
def get_orders():
    try:
        if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
            conn = st.session_state['db_connection']
            cursor = conn.cursor()

            # Join orders with products to display product names
            query = """
            SELECT o.id, p.name, o.quantity 
            FROM orders o
            JOIN products p ON o.product_id = p.id
            ORDER BY o.id DESC
            """

            cursor.execute(query)
            result = cursor.fetchall()

            # Create DataFrame
            df = pd.DataFrame(result, columns=["Order ID", "Product", "Quantity"])
            return df
        else:
            return None
    except Error as e:
        st.error(f"Error retrieving orders: {e}")
        return None


# Check database connection status
if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
    if st.session_state['db_connection'].is_connected():
        st.success("Connected to database")

        # Get products for dropdown
        products = get_products()

        if products and len(products) > 0:
            # Create order form
            st.subheader("Create New Order")
            with st.form("order_form"):
                # Product dropdown
                product_name = st.selectbox("Select Product", options=list(products.keys()))

                # Get product ID from selected name
                product_id = products[product_name]

                # Quantity input
                quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

                # Submit button
                submit_button = st.form_submit_button("Place Order")

                if submit_button:
                    success, message = add_order(product_id, quantity)
                    if success:
                        st.success(message)
                        # Refresh the page to show updated orders
                        st.rerun()
                    else:
                        st.error(message)

            # Display existing orders
            st.subheader("Recent Orders")
            orders_df = get_orders()

            if orders_df is not None and not orders_df.empty:
                st.dataframe(orders_df)
            elif orders_df is not None and orders_df.empty:
                st.info("No orders found in the database.")
            else:
                st.warning("Failed to retrieve orders.")
        else:
            st.warning("No products available. Please add products on Page 1 before creating orders.")
    else:
        st.error("Database connection was established but is no longer active")
else:
    st.warning("No database connection available. Return to main page to connect.")

st.markdown("---")
st.info("Use this form to create new orders by selecting a product and specifying the quantity.")
