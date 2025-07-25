from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import openai
import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import logging
import re


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Worker:
    """Клиент для работы с MCP сервером и OpenAI API."""

    def __init__(self, server_directory: str = "C:/Users/dimaa/PycharmProjects/agent_15_02/mcp_test/telegram-mcp"):
        self.server_params = StdioServerParameters(
            command="uvx",
            args=[
                "browser-use[cli]", "--mcp"
            ],
        )

        self.openai_client = openai.OpenAI(
            api_key=os.getenv("GEMINI_TOKEN"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        # Для переиспользования соединения
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._stdio_context = None
        self._cached_tools: Optional[List[Dict[str, Any]]] = None
        self.messages = []

    async def _ensure_connection(self):
        """Обеспечивает активное соединение с MCP сервером."""
        if self._session is None:
            logger.info("🔌 Устанавливаем соединение с MCP сервером...")
            self._stdio_context = stdio_client(self.server_params)
            self._read, self._write = await self._stdio_context.__aenter__()
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
            await self._session.initialize()
            logger.info("✅ MCP сессия инициализирована.")

    async def get_tools(self):
        """Получает и кэширует список инструментов."""
        if self._cached_tools is None:
            await self._ensure_connection()
            mcp_tools = await self._session.list_tools()
            self._cached_tools = self._format_mcp_tools_for_openai(mcp_tools)
            logger.info(f"✅ Загружено {len(self._cached_tools)} инструментов для ИИ.")
        return self._cached_tools

    def _format_mcp_tools_for_openai(self, mcp_tools):
        """Конвертирует список инструментов из MCP в формат для OpenAI API."""
        formatted_tools = []
        mcp_tools = mcp_tools.tools
        for tool in mcp_tools:
            # Убедимся, что параметры - это словарь, как требует OpenAI
            parameters = tool.inputSchema
            if not isinstance(parameters, dict):
                parameters = {}  # Заглушка, если формат некорректен

            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters
                }
            })
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": "ask_user_for_clarification",
                    "description": "Задает вопрос пользователю для получения уточнений, если вы застряли, инструмент постоянно выдает ошибку или вам нужна дополнительная информация для продолжения.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Конкретный вопрос, который вы хотите задать пользователю."
                            }
                        },
                        "required": ["question"]
                    }
                }
            })
        return formatted_tools

    async def extract_mcp_content(self, tool_result):
        pattern = re.compile(r"text='(.*?)', annotations=", re.DOTALL)
        match = pattern.search(tool_result)
        if match:
            json_string = match.group(1)
            try:
                # Преобразуем чистую JSON-строку в словарь Python [5, 6]
                data = json.loads(json_string)
                return data
            except json.JSONDecodeError as e:
                return "Some problems in extracting tool response content"
        else:
            return "Some problems in extracting tool response content"

    async def close(self):
        """Закрывает соединение с MCP сервером."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Ошибка при закрытии сессии: {e}")

        if self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Ошибка при закрытии stdio контекста: {e}")

        self._session = None
        self._read = None
        self._write = None
        self._stdio_context = None
        self._cached_tools = None
        logger.info("🔌 MCP соединение закрыто.")

    async def run(self, user_prompt):
        """Обрабатывает запрос пользователя, используя постоянное соединение."""
        try:
            # Убеждаемся, что соединение активно
            await self._ensure_connection()

            # Получаем инструменты (из кэша, если уже загружены)
            openai_tools = await self.get_tools()

            # --- Шаг 1: Запрос пользователя ---
            print(f"\n💬 Пользователь: {user_prompt}")
            self.messages.append({"role": "user", "content": user_prompt})

            while True:
                # --- Шаг 2: Первый вызов OpenAI для выбора инструмента ---
                print("🤖 OpenAI выбирает инструмент...")
                response = self.openai_client.chat.completions.create(
                    model="gemini-2.5-flash-lite-preview-06-17",
                    messages=self.messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                response_message = response.choices[0].message
                self.messages.append(response_message)  # Добавляем ответ ИИ в историю


                # --- Шаг 3: Проверяем, выбрал ли ИИ инструмент ---
                if response_message.tool_calls:
                    print("✅ OpenAI выбрал инструмент. Выполняем...")
                    executed_actions = []
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        # Аргументы приходят в виде строки JSON, парсим их
                        function_args = json.loads(tool_call.function.arguments)


                        print(f"Вызов MCP: {function_name}({function_args})")

                        if function_name == "ask_user_for_clarification":
                            question = function_args.get("question", "У меня возникли трудности. Можете помочь?")
                            print(f"\n🤔 Ассистент спрашивает: {question}")
                            self.messages.append(response_message)
                            user_answer = input("Ваш ответ: ")
                            tool_result = f"Пользователь ответил: '{user_answer}'"
                            executed_actions.append(f"ask_user_for_clarification: {question}")

                        else:
                            # --- Шаг 4: Выполняем реальный MCP инструмент ---
                            tool_result = await self._session.call_tool(function_name, function_args)
                            executed_actions.append(f"{function_name}({function_args})")
                            if tool_result and isinstance(tool_result, object):
                                # Достаем JSON-строку, как в нашем самом первом обсуждении
                                tool_result = tool_result.content[0].text
                                try:
                                    tool_result = json.loads(tool_result)
                                    screenshot = tool_result.pop("screenshot", None)
                                    if screenshot:
                                        print("screenshot saved")
                                except:
                                    pass
                                print(tool_result)




                        # --- Шаг 5: Отправляем результат обратно в OpenAI ---
                        print("Отправляем результат выполнения обратно в OpenAI...")
                        self.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(tool_result),
                        })
                        print(self.messages)

                    return executed_actions

                    # --- Шаг 6: Второй вызов OpenAI для финального ответа ---
                else:
                    # Если ИИ ответил сразу, без инструментов
                    print("\n💡 Ответ от OpenAI (без вызова инструментов):")
                    # print(self.messages)
                    print(response_message.content)
                    return response_message.content

        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса: {e}")
            # Если произошла ошибка, сбрасываем соединение для повторной инициализации
            await self.close()
            raise

    def reset_conversation(self):
        """Очищает историю сообщений."""
        self.messages = []
        print("🔄 История разговора очищена.")
