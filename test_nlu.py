import asyncio
from core.nlu import NLUEngine
from core.dispatcher import Dispatcher
from tools.web.answer import WebAnswerTool
from tools.agent.fallbacks import UnknownTool
import os
import sys

def _configure_logging():
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from dotenv import load_dotenv
load_dotenv()
_configure_logging()

async def main():
    print(f"API KEY: {bool(os.environ.get('GROQ_API_KEY'))}")
    d = Dispatcher(tools=[WebAnswerTool(), UnknownTool()], tool_timeout_seconds=5.0)
    
    nlu = NLUEngine(dispatcher=d)
    res = await nlu.infer("What is the capital of France?", request_id="123")
    print("RESULT:", res)

if __name__ == "__main__":
    asyncio.run(main())
