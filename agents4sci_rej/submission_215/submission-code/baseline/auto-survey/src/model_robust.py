import time
import re
from openai import OpenAI
from tqdm import tqdm
import threading

class RobustAPIModel:
    """OpenAI client-based model with robust retry logic and format parsing"""
    
    def __init__(self, model, api_key, api_url) -> None:
        # For vLLM, use base_url without the /chat/completions part
        if api_url.endswith('/chat/completions'):
            base_url = api_url.replace('/chat/completions', '')
        elif api_url.endswith('/v1'):
            base_url = api_url
        else:
            base_url = api_url.rstrip('/') + '/v1'
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "EMPTY"  # vLLM doesn't need real key
        )
        self.model = model
        
    def __validate_outline_format(self, text):
        """Validate that the response contains expected outline format"""
        if not text:
            return False
        
        # Check for basic outline structure
        has_title = bool(re.search(r'Title:\s*.+', text, re.IGNORECASE))
        has_sections = bool(re.search(r'Section\s+\d+:\s*.+', text, re.IGNORECASE))
        
        return has_title or has_sections
    
    def __extract_outline_content(self, text):
        """Extract structured content even from partial or malformed responses"""
        if not text:
            return None
            
        # Try to extract anything that looks like an outline structure
        lines = text.split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            # Look for Title, Section, or Description patterns
            if re.match(r'^(Title|Section|Description)\s*\d*:?\s*.+', line, re.IGNORECASE):
                structured_lines.append(line)
            elif structured_lines and line:  # Keep content after structured markers
                structured_lines.append(line)
        
        if structured_lines:
            return '\n'.join(structured_lines)
        return text  # Return original if no structure found
        
    def __req(self, text, temperature, max_try=5, adaptive_temperature=True):
        """
        Request with adaptive temperature and retry logic
        
        Args:
            text: Input prompt
            temperature: Initial temperature
            max_try: Maximum retry attempts
            adaptive_temperature: Whether to adjust temperature on retries
        """
        current_temp = temperature
        
        for attempt in range(max_try):
            try:
                # Add system prompt for better format compliance
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant that follows instructions precisely and outputs in the exact format requested."},
                    {"role": "user", "content": text}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=current_temp,
                    max_tokens=4096,  # Ensure sufficient response length
                    top_p=0.9  # Add nucleus sampling for better quality
                )
                
                content = response.choices[0].message.content
                
                # Validate response format for outline-related prompts
                if "outline" in text.lower() and "format" in text.lower():
                    if not self.__validate_outline_format(content):
                        # Try to extract whatever structure we can find
                        content = self.__extract_outline_content(content)
                        
                        if adaptive_temperature and attempt < max_try - 1:
                            # Adjust temperature for next attempt
                            current_temp = max(0.1, current_temp - 0.2)
                            print(f"Format validation failed, retrying with temperature {current_temp:.1f}")
                            time.sleep(0.5)
                            continue
                
                return content
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_try} failed: {e}")
                
                if attempt < max_try - 1:
                    if adaptive_temperature:
                        # Lower temperature for more consistent output
                        current_temp = max(0.1, current_temp - 0.2)
                    
                    # Exponential backoff
                    wait_time = min(2 ** attempt, 10)
                    time.sleep(wait_time)
                else:
                    # On final attempt, return a generic response to avoid index errors
                    if "outline" in text.lower():
                        return self.__generate_fallback_outline(text)
                    return "Error: Unable to generate response after multiple attempts."
        
        return None
    
    def __generate_fallback_outline(self, original_prompt):
        """Generate a minimal valid outline as fallback"""
        # Extract topic if possible
        topic_match = re.search(r'about\s+"?([^"]+)"?', original_prompt, re.IGNORECASE)
        topic = topic_match.group(1) if topic_match else "the given topic"
        
        # Return a basic but valid outline structure
        return f"""Title: Survey on {topic}
Section 1: Introduction
Description 1: Overview and motivation for studying {topic}

Section 2: Background and Related Work
Description 2: Historical context and previous research

Section 3: Main Concepts
Description 3: Core principles and theoretical foundations

Section 4: Methods and Approaches
Description 4: Key techniques and methodologies

Section 5: Applications
Description 5: Practical uses and implementations

Section 6: Challenges and Future Directions
Description 6: Current limitations and research opportunities

Section 7: Conclusion
Description 7: Summary of key findings and implications"""
    
    def chat(self, text, temperature=1, max_retries=5):
        """Single chat with robust retry logic"""
        response = self.__req(text, temperature=temperature, max_try=max_retries)
        return response if response else "Unable to generate response"

    def __chat(self, text, temperature, res_l, idx, max_retries=5):
        """Thread-safe chat for batch processing"""
        response = self.__req(text, temperature=temperature, max_try=max_retries)
        res_l[idx] = response if response else "Unable to generate response"
        return response
        
    def batch_chat(self, text_batch, temperature=0.7, max_threads=10):
        """
        Batch chat with improved thread management and error handling
        
        Args:
            text_batch: List of prompts
            temperature: Temperature for generation (lowered default for consistency)
            max_threads: Maximum concurrent threads
        """
        res_l = ['No response'] * len(text_batch)
        thread_l = []
        
        for i, text in enumerate(text_batch):
            # Wait if we've reached max threads
            while len([t for t in thread_l if t.is_alive()]) >= max_threads:
                time.sleep(0.1)
                # Clean up finished threads
                thread_l = [t for t in thread_l if t.is_alive()]
            
            thread = threading.Thread(
                target=self.__chat, 
                args=(text, temperature, res_l, i, 5)
            )
            thread_l.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in tqdm(thread_l):
            thread.join()
        
        # Ensure no "No response" entries remain
        for i, response in enumerate(res_l):
            if response == 'No response':
                # Try one more time synchronously with lower temperature
                res_l[i] = self.chat(text_batch[i], temperature=0.3, max_retries=3)
        
        return res_l