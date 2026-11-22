import asyncio
from core.nlu import NLUEngine
from core.dispatcher import Dispatcher
from tools.web.answer import WebAnswerTool
from tools.agent.fallbacks import UnknownTool

async def main():
    d = Dispatcher(tools=[WebAnswerTool(), UnknownTool()], tool_timeout_seconds=5.0)
    nlu = NLUEngine(dispatcher=d)
    print(nlu._get_system_prompt())

if __name__ == "__main__":
    asyncio.run(main())
