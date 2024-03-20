import streamlit as st
from PIL import Image
import os
from Inception.query import Nearest_images
from Inception.visualFeatureExtract import returnVisualFeatures, Visualiser


def perform_search(image_query, text_query):
    # Placeholder function to perform multimodal search
    st.write("Performing multimodal search...")
    return Nearest_images(image_query, text_query)


def display_search_results(images):
    # Placeholder function to display actual search results
    # For now, displaying sample images as cards
    # images = [
    #     'datasets/DeepFashion/images\\MEN-Jackets_Vests-id_00000094-04_1_front.jpg',
    #     'datasets/DeepFashion/images\\MEN-Denim-id_00001198-01_7_additional.jpg',
    #     "https://via.placeholder.com/300",
    #     "https://via.placeholder.com/300",
    #     "https://via.placeholder.com/300",
    #     "https://via.placeholder.com/300",
    #     "https://via.placeholder.com/300",
    #     "https://via.placeholder.com/300",
    # ]

    num_results = len(images)
    num_columns = 3
    rows = num_results // num_columns + \
        (1 if num_results % num_columns > 0 else 0)

    for i in range(rows):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            idx = i * num_columns + j
            if idx < num_results:
                with cols[j]:
                    st.image(images[idx], caption=f"Result {idx+1}", width=150)

# Main function for the Streamlit app


def main():
    # Set page title and layout
    st.set_page_config(page_title="Multimodal Search App", layout="wide")

    # Page title and description
    st.title("Multimodal Search App")
    st.markdown(
        "This app allows you to perform multimodal search by providing an image and/or text query."
    )

    # Sidebar for user inputs
    st.sidebar.title("Input Queries")

    # Image upload option
    st.sidebar.subheader("Image Query")
    image_query = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"]
    )

    # Text input option
    st.sidebar.subheader("Text Query")
    text_query = st.sidebar.text_input("Enter Text Query")

    # Button to trigger search
    search_button = st.sidebar.button("Search")

    # Display uploaded image, if any
    if image_query is not None:
        st.image(image_query, caption="Uploaded Image", width=300)

    # Display text query, if any
    if text_query:
        st.write("Text Query:", text_query)

    # Display search results, if any (placeholder)
    st.markdown("### Search Results")

    # Perform search when button is clicked
    if search_button and image_query and text_query:
        images = perform_search(image_query, text_query)
        display_search_results(images)


# Run the app
if __name__ == "__main__":
    main()
