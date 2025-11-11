import os
import asyncio
import logging
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomOutputOptions
from livekit.plugins import openai

from interrupt_filter import InterruptFilter, ASREvent

# -----------------------------------------------------------------------------
# Env & logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("realtime-with-tts")
logger.setLevel(logging.INFO)

# Load .env from the folder where this file lives
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(HERE, ".env"))

# If you still see env issues, uncomment these two lines to force set:
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
# os.environ["OPENAI_ORG_ID"] = os.environ.get("OPENAI_ORG_ID", "")

# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.realtime.RealtimeModel(
                model="gpt-4o-realtime-preview",
                modalities=["text"]
            ),  # <-- Added missing comma here to separate kwargs

            tts=openai.TTS(
                model="gpt-4o-mini-tts",
                voice="alloy"
            ),

        )

    @function_tool
    async def get_weather(self, location: str):
        """Called when the user asks about the weather."""
        logger.info(f"Getting weather for {location}")
        return f"The weather in {location} is sunny and the temperature is 20°C."

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    logger.info("🚀 Agent starting...")

    # Interrupt filter
    interrupt_filter = InterruptFilter(logger=logger)

    # Create session
    session = AgentSession()

    # Hook TTS speaking state into filter
    session.on_tts_started = lambda _: interrupt_filter.set_tts_state(True)
    session.on_tts_finished = lambda _: interrupt_filter.set_tts_state(False)

    # What to do on decisions
    @interrupt_filter.on_interrupt
    async def _on_interrupt(decision):
        logger.info(
            f"🟥 User interrupted: {decision.transcript} | reason={decision.reason}")
        # Stop TTS immediately
        await session.tts.stop()

    @interrupt_filter.on_user_speech
    async def _on_user(decision):
        logger.info(f"🟩 User spoke (quiet): {decision.transcript}")

    # Start session (console or room, depends on how CLI started us)
    await session.start(
        agent=WeatherAgent(),
        room=ctx.room,  # None in console mode, a Room in dev/start/connect modes
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=True,
        ),
    )

    # -------------------------------------------------------------------------
    # Transcription wiring
    #   - Console mode -> session.on("transcription")
    #   - Room/dev mode -> session.room.on("transcription")
    # LiveKit requires sync callbacks for `.on()`, so we spawn tasks for async work.
    # -------------------------------------------------------------------------

    def _dispatch_asr(event):
        asyncio.create_task(
            interrupt_filter.handle_asr(
                ASREvent(
                    text=getattr(event, "text", "") or "",
                    is_final=bool(getattr(event, "is_final", False)),
                    alternatives=getattr(event, "alternatives", None),
                    ts_ms=int(getattr(event, "timestamp", 0) * 1000),
                )
            )
        )

    # Console mode (no room)
    @session.on("transcription")
    def _on_session_transcription(event):
        _dispatch_asr(event)

    # Room/dev mode (when room exists)
    room = getattr(session, "room", None)
    if room is not None:
        @room.on("transcription")
        def _on_room_transcription(event):
            _dispatch_asr(event)

    # Kick off a greeting so you can test barge-in
    await session.generate_reply(instructions="Say hello to the user in English")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
