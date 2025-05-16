# pages/1_Page_1.py
import streamlit as st

st.set_page_config(
    page_title="Page 1",
    page_icon="📘",
    layout="wide"
)

st.title("Page 1")
st.write("This is the first additional page of our multipage application.")

st.markdown("""
## Content for Page 1

This page can contain any static text or information you want to display.

### Example Section
- This could be documentation
- Or information about a specific topic
- Or any other static content you need
""")

st.markdown("---")
st.write("Navigate using the sidebar to move between pages.")