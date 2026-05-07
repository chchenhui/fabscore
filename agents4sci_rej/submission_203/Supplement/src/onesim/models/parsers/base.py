from ..core.model_response import ModelResponse


class ParserBase:
    """
    Base class for model response parsers.
    
    Parsers extract structured data from model responses based on 
    specific formats or patterns.
    """
    
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Parse the response and return an updated ModelResponse.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with parsed data.
        """
        raise NotImplementedError("Subclasses must implement parse method")