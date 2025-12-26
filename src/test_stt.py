
import asyncio
import os
import signal
import sys

# Add src to path to allow imports if running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pyaudio
except ImportError:
    print("PyAudio is required for microphone input. Please install it with 'pip install pyaudio'")
    sys.exit(1)

from dotenv import load_dotenv
from elevenlabs_stt import ElevenLabsSTT
from events import STTChunkEvent, STTOutputEvent

# Load environment variables
load_dotenv()

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 2048

async def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not found in environment variables.")
        return

    print("Initializing ElevenLabs STT...")
    stt = ElevenLabsSTT(api_key=api_key, sample_rate=RATE)
    
    # PyAudio setup
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"Failed to open microphone: {e}")
        return

    print("\nListening... (Press Ctrl+C to stop)")
    print("-" * 50)

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    running = True

    async def mic_reader():
        """Reads audio from microphone and sends to STT"""
        while running:
            try:
                data = await loop.run_in_executor(None, stream.read, CHUNK_SIZE, False)
                await stt.send_audio(data)
            except Exception as e:
                print(f"Mic Error: {e}")
                break
        
    async def event_processor():
        """Processes events from STT"""
        async for event in stt.receive_events():
            if isinstance(event, STTChunkEvent):
                print(f"\rPartial: {event.transcript}", end="", flush=True)
            elif isinstance(event, STTOutputEvent):
                print(f"\rFinal:   {event.transcript}")

    # Handle Ctrl+C
    def signal_handler():
        nonlocal running
        print("\nStopping...")
        running = False
        stt_task.cancel()
        mic_task.cancel()

    try:
        mic_task = asyncio.create_task(mic_reader())
        stt_task = asyncio.create_task(event_processor())
        
        await asyncio.gather(mic_task, stt_task)
    except asyncio.CancelledError:
        pass
    finally:
        await stt.close()
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        # Windows specific event loop policy
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
