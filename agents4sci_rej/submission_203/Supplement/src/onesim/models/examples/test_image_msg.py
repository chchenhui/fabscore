"""
Example of using the Message class with image support.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from onesim.models import get_model_manager, get_model, UserMessage, SystemMessage


async def test_image_message():
    """Test the image message functionality."""
    
    # Initialize the model adapter
    manager = get_model_manager()
    manager.load_model_configs("src/onesim/models/examples/my_model_config.json")
    model = get_model("model_1")

    system_msg = SystemMessage(
        content="You are a helpful assistant that can see images."
    )
    
    # Example 1: Using the images parameter
    message1 = UserMessage(
        "What can you see in these images?",
        images=[
            # Replace with actual image paths
            "src/onesim/models/examples/test_1.png",
            "src/onesim/models/examples/test_2.png"
        ]
    )
    
    # Example 2: Using content list with image_path type
    message2 = UserMessage(
        content=[
            {
                "type": "text",
                "text": "Describe these images in detail."
            },
            {
                "type": "image_path",
                "path": "src/onesim/models/examples/test_1.png"
            },
            {
                "type": "image_path",
                "path": "src/onesim/models/examples/test_2.png"
            }
        ]
    )
    
    # Example 3: Using helper functions
    message3 = UserMessage(
        content="What objects are in this image?",
        images=["src/onesim/models/examples/test_1.png"]
    )

    # Example 4: Plain text message
    message4 = UserMessage("What is the weather in Tokyo?")
    
    # Print the formatted messages (for demonstration)
    formatted_messages = model.format(system_msg, message1)
    # print("\nFormatted Message:")
    # print(formatted_messages)
    
    # Generate a response
    response = model(formatted_messages)

    # Get the response text
    print(response.text)


if __name__ == "__main__":
    # Run the async function
    asyncio.run(test_image_message())