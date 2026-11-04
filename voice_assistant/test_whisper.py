from speech_to_text import transcribe_audio


audio_path = "audio/patient_test.m4a"

text = transcribe_audio(audio_path)

print("\nPatient said:")
print(text)