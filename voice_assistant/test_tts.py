print("Starting text-to-speech test...")

from text_to_speech import generate_speech

message = "Hello! I am your AI healthcare assistant. How are you feeling today?"

print("Generating speech. Please wait...")

audio_file = generate_speech(message)

print("Speech generated successfully!")
print("Audio saved at:", audio_file)
