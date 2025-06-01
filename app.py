# import library
import streamlit as st
import pandas as pd
import json
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import folium
from streamlit_folium import folium_static

import tiktoken
from langchain.chains import RetrievalQA, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

url = 'https://places.googleapis.com/v1/places:searchText'

def validate_api_key(api_key):
    """Basic validation for Google Maps API key"""
    return api_key and len(api_key) > 30 and api_key.startswith('AIza')

def get_place_data(destination, api_key, query_type, min_rating, circle_center, radius):
    """Generic function to get place data with error handling"""
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.websiteUri,places.location,places.googleMapsUri',
    }
    
    data = {
        'textQuery': query_type,
        'minRating': min_rating,
        'locationBias': {
            "circle": {
                "center": circle_center,
                "radius": radius
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Check for API key errors first
        if response.status_code == 400:
            error_data = response.json()
            if 'API_KEY_INVALID' in str(error_data):
                st.error("Invalid Google Maps API key. Please check your key.")
                return None
        
        response.raise_for_status()
        result = response.json()
        
        if 'places' not in result:
            st.warning(f"No {query_type.lower()} found matching your criteria")
            return pd.DataFrame()
            
        df = pd.json_normalize(result['places'])
        return df
        
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error fetching {query_type.lower()}: {str(e)}")
        return None

def main():
    st.sidebar.title("Travel Recommendation App Demo")

    api_key = st.sidebar.text_input("Enter Google Maps API key:", type="password")
    gemini_api_key = st.sidebar.text_input("Enter Gemini API key:", type="password", value="AIzaSyAyRGe0bMXW55MEFu49z60kkwwDCPNBodo")

    st.sidebar.write('Please fill in the fields below.')
    destination = st.sidebar.text_input('Destination:', key='destination_app')
    min_rating = st.sidebar.number_input('Minimum Rating:', value=4.0, min_value=0.5, max_value=4.5, step=0.5, key='minrating_app')
    radius = st.sidebar.number_input('Search Radius in meter:', value=3000, min_value=500, max_value=50000, step=100, key='radius_app')
    
    if not destination:
        return
        
    if not validate_api_key(api_key):
        st.error("Please enter a valid Google Maps API key")
        st.info("Keys usually start with 'AIza' and are 39 characters long")
        return

    # Get initial coordinates
    with st.spinner(f"Locating {destination}..."):
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'places.location',
        }
        data = {
            'textQuery': destination,
            'maxResultCount': 1,
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if 'places' not in result or not result['places']:
                st.error(f"No location found for '{destination}'")
                return
                
            location = result['places'][0]['location']
            initial_latitude = location['latitude']
            initial_longitude = location['longitude']
            circle_center = {"latitude": initial_latitude, "longitude": initial_longitude}
            
        except Exception as e:
            st.error(f"Error getting location: {str(e)}")
            return

    # Get all place types
    with st.spinner("Finding recommendations..."):
        df_hotel = get_place_data(
            destination, api_key, 
            f'Place to stay near {destination}', 
            min_rating, circle_center, radius
        )
        df_hotel['type'] = 'Hotel' if df_hotel is not None else None
        
        df_restaurant = get_place_data(
            destination, api_key,
            f'Place to eat near {destination}',
            min_rating, circle_center, radius
        )
        df_restaurant['type'] = 'Restaurant' if df_restaurant is not None else None
        
        df_tourist = get_place_data(
            destination, api_key,
            f'Tourist attraction near {destination}',
            min_rating, circle_center, radius
        )
        df_tourist['type'] = 'Tourist' if df_tourist is not None else None

        # Combine all data
        dfs = [df for df in [df_hotel, df_restaurant, df_tourist] if df is not None]
        if not dfs:
            st.error("No places found matching your criteria")
            return
            
        df_place = pd.concat(dfs, ignore_index=True)
        df_place = df_place.sort_values(by=['userRatingCount', 'rating'], ascending=[False, False])
        
        df_place_rename = df_place[[
            'type', 'displayName.text', 'formattedAddress', 'rating', 
            'userRatingCount', 'googleMapsUri', 'websiteUri', 
            'location.latitude', 'location.longitude'
        ]].rename(columns={
            'displayName.text': 'Name',
            'rating': 'Rating',
            'googleMapsUri': 'Google Maps URL',
            'websiteUri': 'Website URL',
            'userRatingCount': 'User Rating Count',
            'location.latitude': 'Latitude',
            'location.longitude': 'Longitude',
            'formattedAddress': 'Address',
            'type': 'Type'
        })

    def database():
        st.dataframe(df_place_rename)

    def maps():
        st.header("🌏 Travel Recommendation App 🌏")
        places_type = st.radio('Looking for:', ["Hotels 🏨", "Restaurants 🍴", "Tourist Attractions ⭐"])
        
        initial_location = [initial_latitude, initial_longitude]
        type_colour = {'Hotel':'blue', 'Restaurant':'green', 'Tourist':'orange'}
        type_icon = {'Hotel':'home', 'Restaurant':'cutlery', 'Tourist':'star'}
        
        df_to_show = {
            "Hotels 🏨": df_hotel,
            "Restaurants 🍴": df_restaurant,
            "Tourist Attractions ⭐": df_tourist
        }.get(places_type, pd.DataFrame())
        
        if df_to_show is None or df_to_show.empty:
            st.warning(f"No {places_type} found")
            return
            
        with st.spinner("Loading map..."):
            for index, row in df_to_show.iterrows():
                location = [row['location.latitude'], row['location.longitude']]
                m = folium.Map(location=initial_location, zoom_start=13)
                
                content = f"""
                <b>{row['displayName.text']}</b><br>
                Rating: {row['rating']}<br>
                {row['formattedAddress']}<br>
                <a href="{row['googleMapsUri']}" target="_blank">View on Maps</a>
                """
                
                folium.Marker(
                    location=location,
                    popup=folium.Popup(content, max_width=300),
                    icon=folium.Icon(
                        color=type_colour[row['type']],
                        icon=type_icon[row['type']]
                    )
                ).add_to(m)
                
                st.write(f"## {index + 1}. {row['displayName.text']}")
                from streamlit_folium import st_folium
                st_folium(m, returned_objects=[])
                st.write(f"Rating: {row['rating']}")
                st.write(f"Address: {row['formattedAddress']}")
                if pd.notna(row['websiteUri']):
                    st.write(f"Website: {row['websiteUri']}")
                st.write(f"[More info]({row['googleMapsUri']})")

    def chatbot():
        class Message(BaseModel):
            actor: str
            payload: str

        if not gemini_api_key:
            st.error("Please enter a valid Gemini API key")
            return
            
        llm = ChatGoogleGenerativeAI(
            google_api_key=gemini_api_key,
            model="gemini-pro",
            temperature=0
        ) 

        if "messages" not in st.session_state:
            st.session_state.messages = [Message(actor="ai", payload="Hi! How can I help you?")]
        
        for msg in st.session_state.messages:
            st.chat_message(msg.actor).write(msg.payload)

        if query := st.chat_input("Enter a prompt here"):
            st.session_state.messages.append(Message(actor="user", payload=query))
            st.chat_message("user").write(query)
            
            with st.spinner("Thinking..."):
                try:
                    # Prepare context
                    df_place['combined_info'] = df_place.apply(
                        lambda row: f"{row['type']}: {row['displayName.text']} (Rating: {row['rating']}) at {row['formattedAddress']}",
                        axis=1
                    )
                    
                    loader = DataFrameLoader(df_place, page_content_column="combined_info")
                    docs = loader.load()
                    
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(docs)
                    
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-l6-v2",
                        model_kwargs={'device':'cpu'},
                        encode_kwargs={'normalize_embeddings': False}
                    )
                    
                    vectorstore = FAISS.from_documents(texts, embeddings)
                    
                    template = """ 
                    You're a travel assistant helping with recommendations in {destination}.
                    Available places:\n{context}\n\n
                    Chat history:\n{history}\n\n
                    Question: {question}\n
                    Provide helpful recommendations from the available places when relevant.
                    """
                    
                    prompt = PromptTemplate(
                        input_variables=["destination", "context", "history", "question"],
                        template=template,
                    )
                    
                    memory = ConversationBufferMemory(
                        memory_key="history",
                        input_key="question",
                        return_messages=True
                    )
                    
                    qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type='stuff',
                        retriever=vectorstore.as_retriever(),
                        chain_type_kwargs={
                            "prompt": prompt,
                            "memory": memory,
                            "verbose": True
                        }
                    )
                    
                    response = qa.run({
                        "destination": destination,
                        "question": query,
                        "context": "\n".join(df_place['combined_info'])
                    })
                    
                    st.session_state.messages.append(Message(actor="ai", payload=response))
                    st.chat_message("ai").write(response)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    method = st.sidebar.radio(" ", ["Search 🔎", "ChatBot 🤖", "Database 📑"], key="method_app")
    if method == "Search 🔎":
        maps()
    elif method == "ChatBot 🤖":
        chatbot()
    else:
        database()

    st.sidebar.markdown(''' 
        ## Created by: 
        Ahmad Luay Adnani - [GitHub](https://github.com/ahmadluay9) 
        ''')

if __name__ == '__main__':
    main()