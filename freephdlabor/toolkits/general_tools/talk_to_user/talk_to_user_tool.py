from smolagents import Tool

class TalkToUser(Tool):
    name = "talk_to_user"
    description = """
    This tool displays a message from the LLM to the user (e.g., a question, statement, or instruction),
    waits for the user's reply, and returns the response.
    """
    inputs = {
        "message": {
            "type": "string",
            "description": "The message to display to the user (question, instruction, etc.)"
        }
    }
    output_type = "string"

    def forward(self, message: str) -> str:
        print(f"[LLM]: {message}")
        user_response = input("[User]: ")
        return user_response