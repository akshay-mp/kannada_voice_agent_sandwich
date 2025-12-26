
import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pyaudio
except ImportError:
    print("PyAudio is required for audio playback. Please install it with 'pip install pyaudio'")
    sys.exit(1)

from dotenv import load_dotenv
from elevenlabs_tts import ElevenLabsTTS
from events import TTSChunkEvent

# Load environment variables
load_dotenv()

# Audio configuration
FORMAT = pyaudio.paInt16 # ElevenLabs returns PCM, likely 16-bit or similar depending on format
CHANNELS = 1
RATE = 24000 # Default for output_format="pcm_24000"

async def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not found in environment variables.")
        return

    print("Initializing ElevenLabs TTS...")
    # Initialize with PCM 24000 format
    tts = ElevenLabsTTS(
        api_key=api_key,
        output_format="pcm_24000" 
    )

    # PyAudio setup
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True
        )
    except Exception as e:
        print(f"Failed to open audio output: {e}")
        await tts.close()
        return

    test_text = "Hello! This is a test of the Eleven Labs text to speech system. I hope it sounds good."
    print(f"\nSending text: '{test_text}'")
    
    async def play_audio():
        print("Waiting for audio...", end="", flush=True)
        async for event in tts.receive_events():
            if isinstance(event, TTSChunkEvent):
                print(".", end="", flush=True)
                # Play audio chunk
                await asyncio.to_thread(stream.write, event.audio)
        print("\nPlayback finished.")

    try:
        # Start receiving task
        receive_task = asyncio.create_task(play_audio())
        
        # Send text
        await tts.send_text(test_text)
        
        # Wait a bit for playback to finish (simple heuristic for test script)
        # In a real app we'd wait for a "done" signal or manage stream lifecycle better
        await asyncio.sleep(5) 
        
        await tts.close()
        # Cancel receive task if it's still waiting
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
            
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
