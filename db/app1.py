import streamlit as st
import mysql.connector
from mysql.connector import Error

# Set page configuration
st.set_page_config(
    page_title="Simple Multipage App",
    page_icon="📚",
    layout="wide"
)


# Database connection function
def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host="mysql",
            user="root",
            password="rootpassword"
        )
        if connection.is_connected():
            db_info = connection.get_server_info()
            st.session_state['db_connection'] = connection
            return True, f"Connected to MySQL Server version {db_info}"
        else:
            return False, "Failed to connect to database"
    except Error as e:
        return False, f"Error: {e}"


# Main page content
def main():
    st.title("Welcome to the Main Page")
    st.write("This is the main page of our simple multipage Streamlit application.")

    # Initialize session state for database connection if not already present
    if 'db_connection' not in st.session_state:
        st.session_state['db_connection'] = None

    # Database connection button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Connect to Database"):
            success, message = connect_to_mysql()
            if success:
                st.session_state['db_status'] = "connected"
                st.session_state['db_message'] = message
            else:
                st.session_state['db_status'] = "error"
                st.session_state['db_message'] = message

    # Display connection status if available
    if 'db_status' in st.session_state:
        with col2:
            if st.session_state['db_status'] == "connected":
                st.success(st.session_state['db_message'])
            else:
                st.error(st.session_state['db_message'])

    st.markdown("""
    ### Navigation
    Use the sidebar to navigate between pages:
    - **Main Page** (You are here)
    - **Page 1** - Additional content
    - **Page 2** - More information
    """)

    st.markdown("---")
    st.write("This is a simple demonstration of Streamlit's multipage capabilities.")

    # Display database connection status at the bottom
    if st.session_state['db_connection'] is not None and st.session_state['db_connection'].is_connected():
        st.info("Database connection is active and available on all pages")
    else:
        st.warning("Not connected to database. Click the button above to connect.")


# Create the app structure
if __name__ == "__main__":
    main()