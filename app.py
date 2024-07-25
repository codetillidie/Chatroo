import streamlit as st
import random
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
import openai
import json
import time
import os
from google.auth.exceptions import TransportError
import logging
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MAX_TURNS = 5
MAX_RETRIES = 3
RETRY_DELAY = 2
CONVERSATION_STATES = ['START', 'USER_TURN', 'AI_TURN', 'FEEDBACK']

IMAGES_FOLDER_ID = os.getenv('IMAGES_FOLDER_ID')
METADATA_FILE_ID = os.getenv('METADATA_FILE_ID')
SHEET_ID = os.getenv('GOOGLE_SHEET_ID')

# Set up Google Drive and Sheets API clients
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/spreadsheets']

google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if google_creds:
    creds_json = base64.b64decode(google_creds).decode('utf-8')
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
else:
    creds = service_account.Credentials.from_service_account_file(
        'path/to/your/service_account_key.json', scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=creds)
sheets_service = build('sheets', 'v4', credentials=creds)

# OpenAI API setup
openai.api_key = os.getenv('OPENAI_API_KEY')

def retry_with_backoff(func):
    """Decorator for retrying functions with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (TransportError, openai.error.OpenAIError) as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Max retries reached for {func.__name__}: {str(e)}", exc_info=True)
                    raise
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY * (2 ** attempt))
    return wrapper

@retry_with_backoff
def get_metadata():
    """Retrieve the metadata JSON file from Google Drive and parse it."""
    try:
        file = drive_service.files().get(fileId=METADATA_FILE_ID).execute()
        content = drive_service.files().get_media(fileId=METADATA_FILE_ID).execute()
        return json.loads(content.decode('utf-8'))
    except Exception as e:
        logger.error(f"Error retrieving metadata: {str(e)}", exc_info=True)
        return {}

@retry_with_backoff
def get_image_list():
    """Retrieve a list of image files from the specified Google Drive folder."""
    try:
        results = drive_service.files().list(
            q=f"'{IMAGES_FOLDER_ID}' in parents and mimeType contains 'image/'",
            fields="files(id, name, webContentLink)"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        logger.error(f"Error retrieving image list: {str(e)}", exc_info=True)
        return []

def cache_data(data, filename):
    """Cache the data to a local file."""
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_cached_data(filename):
    """Load the cached data from a local file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_random_image_and_scenario():
    """Get a random image and its corresponding scenario description."""
    cached_images = load_cached_data('image_cache.json')
    cached_metadata = load_cached_data('metadata_cache.json')
    
    if not cached_images or not cached_metadata:
        images = get_image_list()
        metadata = get_metadata()
        if images and metadata:
            cache_data(images, 'image_cache.json')
            cache_data(metadata, 'metadata_cache.json')
        else:
            logger.error("Error retrieving images or metadata.")
            return None, None
    else:
        images = cached_images
        metadata = cached_metadata

    if images:
        random_image = random.choice(images)
        image_url = random_image['webContentLink']
        filename = random_image['name']
        
        scenario = metadata.get(filename, {}).get('description', "No description available.")
        
        return image_url, scenario
    else:
        logger.error("No images available.")
        return None, None

@retry_with_backoff
def generate_ai_response(conversation_history, scenario):
    """Generate AI response using OpenAI API with retry mechanism."""
    messages = [
        {"role": "system", "content": f"You are a person in this scenario: {scenario}. Engage in small talk with the user."},
    ]
    messages.extend([{"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]} for msg in conversation_history])
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()

@retry_with_backoff
def generate_feedback(conversation_history):
    """Generate feedback on the user's conversation skills with retry mechanism."""
    messages = [
        {"role": "system", "content": "You are an expert in conversation skills. Analyze the following conversation and provide feedback on the user's small talk skills. Focus on enthusiasm, humor, positivity, how inviting the conversation is, and its open-endedness."},
        {"role": "user", "content": f"Here's the conversation:\n\n{json.dumps(conversation_history, indent=2)}\n\nProvide feedback on the user's conversation skills."}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()

@retry_with_backoff
def save_conversation_to_sheets(image_url, scenario, conversation_history, feedback):
    """Save the conversation history and feedback to Google Sheets."""
    values = [
        [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            image_url,
            scenario,
            json.dumps(conversation_history),
            feedback
        ]
    ]
    
    body = {
        'values': values
    }
    
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range='Sheet1',
        valueInputOption='RAW',
        insertDataOption='INSERT_ROWS',
        body=body
    ).execute()
    logger.info("Conversation saved to Google Sheets successfully")

def main():
    st.title("Small Talk Trainer")

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
        st.session_state.image_url, st.session_state.scenario = get_random_image_and_scenario()
        st.session_state.state = 'START'
        st.session_state.turns = 0

    # Display image and scenario
    if st.session_state.image_url:
        st.image(st.session_state.image_url, caption="Conversation Scenario")
        st.write(st.session_state.scenario)
    else:
        st.error("Failed to load image and scenario. Please try again.")
        logger.error("Failed to load image and scenario")
        return

    # Display conversation history
    for message in st.session_state.conversation:
        st.write(f"{'You' if message['role'] == 'user' else 'AI'}: {message['content']}")

    # Handle conversation flow
    if st.session_state.state == 'START' or st.session_state.state == 'USER_TURN':
        user_input = st.text_input("Your response:", key=f"user_input_{st.session_state.turns}")
        if st.button("Send"):
            st.session_state.conversation.append({"role": "user", "content": user_input})
            st.session_state.turns += 1
            st.session_state.state = 'AI_TURN'
            logger.info(f"User input: {user_input}")
            st.experimental_rerun()

    elif st.session_state.state == 'AI_TURN':
        try:
            ai_response = generate_ai_response(st.session_state.conversation, st.session_state.scenario)
            st.session_state.conversation.append({"role": "assistant", "content": ai_response})
            logger.info(f"Generated AI response: {ai_response}")
            
            if st.session_state.turns >= MAX_TURNS:
                st.session_state.state = 'FEEDBACK'
            else:
                st.session_state.state = 'USER_TURN'
            st.experimental_rerun()
        except Exception as e:
            logger.error(f"Failed to generate AI response: {str(e)}", exc_info=True)
            st.error("Failed to generate AI response. Please try again.")
            st.session_state.state = 'USER_TURN'

    elif st.session_state.state == 'FEEDBACK':
        try:
            feedback = generate_feedback(st.session_state.conversation)
            st.write("Conversation ended. Here's your feedback:")
            st.write(feedback)
            logger.info(f"Generated feedback: {feedback}")
            
            # Save conversation to Google Sheets
            try:
                save_conversation_to_sheets(
                    st.session_state.image_url,
                    st.session_state.scenario,
                    st.session_state.conversation,
                    feedback
                )
                st.success("Conversation saved successfully!")
            except Exception as e:
                logger.error(f"Failed to save conversation: {str(e)}", exc_info=True)
                st.error("Failed to save conversation. Don't worry, you can still continue.")
            
            if st.button("Start New Conversation"):
                st.session_state.conversation = []
                st.session_state.image_url, st.session_state.scenario = get_random_image_and_scenario()
                st.session_state.state = 'START'
                st.session_state.turns = 0
                logger.info("Started new conversation")
                st.experimental_rerun()
        except Exception as e:
            logger.error(f"Failed to generate feedback: {str(e)}", exc_info=True)
            st.error("Failed to generate feedback. Please try again.")
            if st.button("Try Again"):
                st.experimental_rerun()

if __name__ == "__main__":
    main()
