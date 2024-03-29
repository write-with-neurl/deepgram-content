import os
import json
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    SpeakOptions,
    FileSource,
)


# Load environment variables
load_dotenv()

# Create a `.env` file and add your API Key like so: DG_API_KEY = "YOUR API KEY"

# Next, Import your API_KEY via system variable as best practice.
# Retrieve API Key from environment variables
API_KEY = os.getenv("DG_API_KEY")
if not API_KEY:
    raise ValueError("Please set the DG_API_KEY environment variable.")

#Initialize the API Key with the Deepgram Client-side helper function
deepgram = DeepgramClient(API_KEY)

# Path to the audio file and API Key
AUDIO_FILE = "your_audio_file.m4a"


def get_transcript(payload, options):
    """
    Returns a JSON of Deepgram's transcription given an audio file.
    """
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    return json.loads(response.to_json(indent=4))


def get_topics(transcript):
    """
    Returns back a list of all unique topics in a transcript.
    """
    topics = set()  # Initialize an empty set to store unique topics

    # Traverse through the JSON structure to access topics
    for segment in transcript['results']['topics']['segments']:
        # Iterate over each topic in the current segment
        for topic in segment['topics']:
            # Add the topic to the set
            topics.add(topic['topic'])
    return topics


def get_summary(transcript):
    """
    Returns the summary of the transcript as a string.
    """
    return transcript['results']['summary']['short']


def save_speech_summary(transcript, options):
    """
    Writes an audio summary of the transcript to disk.
    """
    s = {"text": get_summary(transcript)}
    filename = "output.wav"
    response = deepgram.speak.v("1").save(filename, s, options)


def main():
    try:
        # STEP 1: Ingest the audio file
        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        #STEP 2: Configure Deepgram options for audio analysis
        text_options = PrerecordedOptions(
            model="nova-2",
            language="en",
            summarize="v2", 
            topics=True, 
            intents=True, 
            smart_format=True, 
            sentiment=True, 
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        r = get_transcript(payload, text_options)

        # STEP 4: Print responses that can be used for integration with an app
        print('Topics:', get_topics(r))
        print('Summary:', get_summary(r))

        # STEP 5: Additionally, these summaries can also be spoken back to you
        speak_options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )
        save_speech_summary(r, speak_options)

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()
