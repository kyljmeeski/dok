from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Load from Hugging Face Hub
processor = AutoProcessor.from_pretrained("UlutSoftLLC/whisper-small-kyrgyz")
model = AutoModelForSpeechSeq2Seq.from_pretrained("UlutSoftLLC/whisper-small-kyrgyz")

# Save to local directory
save_path = "./whisper-small-kyrgyz-local"
processor.save_pretrained(save_path)
model.save_pretrained(save_path)
