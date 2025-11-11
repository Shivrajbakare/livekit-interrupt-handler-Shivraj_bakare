import logging
import re
from collections.abc import AsyncIterable as _AsyncIterable

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage

from livekit.agents.llm import function_tool

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You are a helpful assistant. Keep responses concise.",
        )


    async def stt_node(self, audio, model_settings):
        """
        Custom STT processing node that filters out filler words before emitting
        final transcription events. It wraps the default agent STT node and
        checks final transcripts for unwanted fillers.
        """

        stt_stream = Agent.default.stt_node(self, audio, model_settings)

        ignore_tokens = ["umm", "um", "uh", "haan", "okay", "understood", "continue"]

        async for evt in stt_stream:
            if evt.type == "final_transcript":
                transcript = evt.alternatives[0].text.lower().strip()

                if any(word in transcript for word in ignore_tokens):
                    print("Filtered filler phrase:", transcript)
                    continue

                yield evt

        yield stt_stream



    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    logger.info("Starting simplified agent session...")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        preemptive_generation=False,
        allow_interruptions=True
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
