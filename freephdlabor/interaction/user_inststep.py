from smolagents.memory import TaskStep
from smolagents.models import ChatMessage, MessageRole

class UserInstructionStep(TaskStep):
    user_instruction: str
    user_images: list["PIL.Image.Image"] | None = None

    def __init__(self, user_instruction: str, user_images: list["PIL.Image.Image"] | None = None, **kwargs):
        # Initialize parent with a dummy task or skip if TaskStep allows it
        super().__init__(task=user_instruction, task_images = user_images, **kwargs)
        self.user_instruction = user_instruction
        self.user_images = user_images

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"Additional instruction from the user:\n{self.user_instruction}"}]
        if self.user_images:
            content.extend([{"type": "image", "image": image} for image in self.user_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]
  