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


# This is the code that i have added


    async def stt_node(
            self, audio, model_settings):
        """

        A node in the processing pipeline that transcribes audio frames into speech events.

        By default, this node uses a Speech-To-Text (STT) capability from the current agent.
        If the STT implementation does not support streaming natively, a VAD (Voice Activity
        Detection) mechanism is required to wrap the STT.

        You can override this node with your own implementation for more flexibility (e.g.,
        custom pre-processing of audio, additional buffering, or alternative STT strategies).

        Args:
            audio (AsyncIterable[rtc.AudioFrame]): An asynchronous stream of audio frames.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields:
            stt.SpeechEvent: An event containing transcribed text or other STT-related data.
        """

        se = Agent.default.stt_node(self, audio, model_settings)
        async for event in se:
            if event.type == "final_transcript":
                filler_words = ["umm", "uh", "um",
                                "haan", "understood", "okay","continue"]
                text = event.alternatives[0].text.lower()

                if any(filler in text for filler in filler_words):
                    print("found filler word:" , text)
                  
                    print ()

                else:
                    yield event

        yield se

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    logger.info("Starting simplified agent session...")

    session = AgentSession(
        # stt="deepgram/base",
        stt="assemblyai/universal-streaming:en",
        # llm="openai/gpt-3.5-turbo",
        llm="openai/gpt-4.1-mini",
        # tts="elevenlabs/eleven_multilingual_v2",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # Minimal configuration for stability
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
