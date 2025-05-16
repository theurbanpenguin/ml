# pages/2_Page_2.py
import streamlit as st

st.set_page_config(
    page_title="Page 2",
    page_icon="📗",
    layout="wide"
)

st.title("Page 2")
st.write("This is the second additional page of our multipage application.")

# Check database connection status
if 'db_connection' in st.session_state and st.session_state['db_connection'] is not None:
    if st.session_state['db_connection'].is_connected():
        st.success("Database connection is available on this page")
    else:
        st.error("Database connection was established but is no longer active")
else:
    st.warning("No database connection available. Return to main page to connect.")

st.markdown("""
## Content for Page 2

More static content can be placed here.

### Another Section
This page demonstrates how you can organize different information across 
multiple pages for better user experience.

You can add as much text as you need here.
""")

st.markdown("---")
st.info("This is the last page of our simple multipage app.")