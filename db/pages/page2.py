# pages/2_Page_2.py
import streamlit as st

st.set_page_config(
    page_title="Page 2",
    page_icon="📗",
    layout="wide"
)

st.title("Page 2")
st.write("This is the second additional page of our multipage application.")

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