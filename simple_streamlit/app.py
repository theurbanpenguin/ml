import streamlit as st

# Set page title
st.set_page_config(page_title="My First Streamlit App")

# Display a title
st.title("Welcome to Streamlit!")

# Display some text
st.write("This is a simple Streamlit application.")

# Display markdown
st.markdown("## Features of Streamlit")
st.markdown("* Easy to use")
st.markdown("* Quick to deploy")
st.markdown("* Great for data applications")

# Display a subheader
st.subheader("About This App")
st.text("This is a demonstration of basic Streamlit functionality.")