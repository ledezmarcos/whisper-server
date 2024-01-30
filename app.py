# import whisper

# model = whisper.load_model("base")
# result = model.transcribe("test2.mp3", language="en")
# print(result["text"])
from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

# Load Whisper model (consider doing this outside the request handling to save loading time)
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if the request contains an audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_filename = "temp_audio_file.m4a"
    audio_file.save(audio_filename)

    # Transcribe the audio file using Whisper
    try:
        result = model.transcribe(audio_filename, language="en")
        transcription_text = result['text']
    except Exception as e:
        # Handle transcription errors
        return jsonify({'error': 'Failed to transcribe audio', 'details': str(e)}), 500
    finally:
        # Cleanup: remove the temporary audio file
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

    # Return the transcription result
    return jsonify({'transcription': transcription_text}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

