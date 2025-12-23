import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import datetime

# --- Data Structures ---

@dataclass
class Message:
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"role": self.role, "content": self.content}

# --- Interfaces ---

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        pass

class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Returns JSON schema for the tool arguments."""
        pass

# --- Implementations ---

class CalculatorTool(Tool):
    name = "calculator"
    description = "Useful for performing basic arithmetic calculations. Input should be a mathematical expression string."

    def execute(self, expression: str) -> str:
        try:
            # WARNING: eval is dangerous in production, using for simple demo purposes only.
            # In a real system, use a safe math parser.
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
               return "Error: Invalid characters in expression."
            return str(eval(expression))
        except Exception as e:
            return f"Error executing calculation: {e}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2')"
                }
            },
            "required": ["expression"]
        }

class MockLLMProvider(LLMProvider):
    """
    A mock LLM that follows a simple script for demonstration purposes.
    It simulates a "Think -> Call Tool -> Response" cycle.
    """
    def generate(self, messages: List[Message]) -> str:
        last_msg = messages[-1]
        
        # Simple heuristic to simulate intelligence
        if last_msg.role == "user":
            if "calculate" in last_msg.content.lower():
                # Simulate thinking and deciding to use a tool
                return json.dumps({
                    "thought": "The user wants to calculate something. I should use the calculator tool.",
                    "tool": "calculator",
                    "tool_input": {"expression": "25 * 4"} 
                }) # Note: Hardcoded for demo, normally would extract numbers
            else:
                return "I am a simple agent. I can help you with calculations."
        
        elif last_msg.role == "tool":
            return "The result of the calculation is " + last_msg.content
            
        return "I don't know what to do."

class Memory:
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, message: Message):
        self.messages.append(message)

    def get_history(self) -> List[Message]:
        return self.messages

class Agent:
    def __init__(self, llm: LLMProvider, tools: List[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory = Memory()
        self.system_prompt = "You are a helpful AI assistant with access to tools."

    def run(self, user_query: str):
        print(f"\n--- User: {user_query} ---")
        self.memory.add_message(Message(role="user", content=user_query))

        # 1. Think / Decide (LLM Call)
        response_text = self.llm.generate(self.memory.get_history())
        
        try:
            # Attempt to parse as JSON (structural thinking)
            decision = json.loads(response_text)
            print(f"[Agent Thought]: {decision.get('thought')}")
            
            # 2. Act (Tool Execution)
            tool_name = decision.get("tool")
            if tool_name and tool_name in self.tools:
                tool_input = decision.get("tool_input", {})
                print(f"[Agent Action]: Calling tool '{tool_name}' with {tool_input}")
                
                tool_result = self.tools[tool_name].execute(**tool_input)
                print(f"[Tool Output]: {tool_result}")
                
                # 3. Observe (Update Memory & Final Response)
                self.memory.add_message(Message(role="tool", content=tool_result, metadata={"tool": tool_name}))
                
                # Final LLM call to synthesize answer
                final_response = self.llm.generate(self.memory.get_history())
                print(f"[Agent Response]: {final_response}")
                self.memory.add_message(Message(role="assistant", content=final_response))
            else:
                 print(f"[Agent Response]: {response_text}")
        
        except json.JSONDecodeError:
            # Fallback for simple text response
            print(f"[Agent Response]: {response_text}")
            self.memory.add_message(Message(role="assistant", content=response_text))

# --- Main Execution ---

if __name__ == "__main__":
    # Setup
    llm = MockLLMProvider()
    calc_tool = CalculatorTool()
    agent = Agent(llm=llm, tools=[calc_tool])

    # Run Scenarios
    agent.run("Hello, who are you?")
    agent.run("Please calculate 25 * 4 for me.")
