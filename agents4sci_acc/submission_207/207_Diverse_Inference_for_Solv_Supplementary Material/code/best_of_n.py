import logging
import os
import time
import random
import json
from datetime import datetime
from anthropic import Anthropic
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def save_bon_results(result_data, model_name, test_id=None):
    """
    Save Best-of-N sampling results to a JSON file for analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs("output/bon_results", exist_ok=True)
    
    # Generate a timestamp and ID for the file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_id = test_id or timestamp
    
    # Create the filename
    filename = f"output/bon_results/{model_name.replace('/', '_')}_{test_id}.json"
    
    # Save the data
    with open(filename, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    logger.info(f"Saved BoN results to {filename}")
    return filename


def best_of_n_sampling(system_prompt: str, initial_query: str, client, model: str, n: int = 8) -> str:
    """
    Best-of-N sampling for Anthropic, OpenAI or Gemini API
    """
    bon_completion_tokens = 0
    
    # Check which client we're using
    if hasattr(client, "__class__") and client.__class__.__name__ == "Anthropic":
        return _best_of_n_sampling_anthropic(system_prompt, initial_query, client, model, n)
    elif model.startswith("gemini-"):
        return _best_of_n_sampling_gemini(system_prompt, initial_query, client, model, n)
    else:
        return _best_of_n_sampling_openai(system_prompt, initial_query, client, model, n)


def _best_of_n_sampling_anthropic(system_prompt: str, initial_query: str, client, model: str, n: int = 8) -> str:
    """
    Best-of-N sampling implementation for Anthropic API
    """
    bon_completion_tokens = 0

    # Format messages for Anthropic API
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{initial_query}"}
    ]

    completions = []
    
    # Generate n completions
    for _ in range(n):
        response = client.messages.create(
            model=model,
            messages=messages,
            max_tokens=4096
        )
        completions.append(response.content[0].text)
        # Anthropic doesn't provide token usage in the same way as OpenAI
        # We could implement our own token counter if needed
    
    logger.info(f"Generated {len(completions)} initial completions")
    
    # Rate the completions
    ratings = []
    for completion in completions:
        rating_prompt = f"{system_prompt}\n\n{initial_query}\n\nRate the following response on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness. Respond with only a number.\n\n{completion}"
        
        rating_response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": rating_prompt}],
            max_tokens=10
        )
        
        try:
            rating = float(rating_response.content[0].text.strip())
            ratings.append(rating)
        except ValueError:
            ratings.append(0)
    
    best_index = ratings.index(max(ratings))
    return completions[best_index]


def _best_of_n_sampling_openai(system_prompt: str, initial_query: str, client, model: str, n: int = 3) -> str:
    """
    Best-of-N sampling implementation for OpenAI API
    """
    bon_completion_tokens = 0

    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": initial_query}]

    completions = []

    # Check if this is gpt-4o-search-preview model
    if model == "gpt-4o-search-preview":
        # Special handling for search-enabled model
        # Make multiple individual requests since n parameter isn't supported
        num_requests = 8  # Make 8 requests, similar to n=8 in standard implementation
        
        for i in range(num_requests):
            # For search models, we can't modify the temperature, so we'll add a small
            # note to encourage some variation in responses
            modified_messages = messages.copy()
            if i > 0:  # Only modify messages for subsequent requests to get some variation
                modified_messages.append({
                    "role": "user", 
                    "content": f"I'd like a slightly different perspective on this. (variation #{i})"
                })
            
            response = client.chat.completions.create(
                model=model,
                web_search_options={"search_context_size": "high"},
                messages=modified_messages
            )
            completions.append(response.choices[0].message.content)
            logger.info(f"Generated completion {i+1}/{num_requests} using search-enabled model.")
            if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens'):
                bon_completion_tokens += response.usage.completion_tokens
        
        logger.info(f"Generated {len(completions)} completions with search-enabled model.")
        
        # Continue with rating process instead of returning early
        # Rating will be handled in the common code below
    else:
        # Standard approach for non-search models
        response = client.chat.completions.create(
            model=model,
            # reasoning_effort="high",
            messages=messages,
            n=8,
        )
        completions = [choice.message.content for choice in response.choices]
        logger.info(f"Generated {len(completions)} initial completions. Tokens used: {response.usage.completion_tokens}")
        bon_completion_tokens += response.usage.completion_tokens

    # Rate the completions (common code for both search and non-search models)
    rating_messages = messages.copy()
    rating_messages.append({"role": "developer",
                            "content": "Rate the following responses on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness. Respond with only a number."})

    ratings = []
    for completion in completions:
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "user", "content": "Rate the above response:"})

        # Handle rating differently for search models since they don't support certain parameters
        if model == "gpt-4o-search-preview":
            # For search models, we need to be careful about parameters we pass
            modified_rating_messages = rating_messages.copy()
            
            # We might need to modify the messages slightly to encourage accurate ratings
            # without using temperature or other parameters
            rating_response = client.chat.completions.create(
                model=model,
                web_search_options={"search_context_size": "high"},
                messages=modified_rating_messages,
            )
        else:
            rating_response = client.chat.completions.create(
                model=model,
                # reasoning_effort="high",
                messages=rating_messages,
                n=1,
            )
            
        if hasattr(rating_response, 'usage') and hasattr(rating_response.usage, 'completion_tokens'):
            bon_completion_tokens += rating_response.usage.completion_tokens
            
        try:
            rating = float(rating_response.choices[0].message.content.strip())
            ratings.append(rating)
        except ValueError:
            ratings.append(0)

        rating_messages = rating_messages[:-2]

    best_index = ratings.index(max(ratings))
    return completions[best_index], bon_completion_tokens
    
    
def _best_of_n_sampling_gemini(system_prompt: str, initial_query: str, client, model: str, n: int = 2) -> str:
    """
    Special handling for gemini-2.0-flash-thinking-exp-01-21, which has different response structures
    """
    # Check if we're dealing with the special flash-thinking model
    if "flash-thinking" in model:
        return _best_of_n_sampling_gemini_flash_thinking(system_prompt, initial_query, client, model, n)
    else:
        return _best_of_n_sampling_gemini_standard(system_prompt, initial_query, client, model, n)

def _best_of_n_sampling_gemini_flash_thinking(system_prompt: str, initial_query: str, client, model: str, n: int = 8) -> str:
    """
    Implementation for the gemini-2.0-flash-thinking-exp-01-21 model which has a different response structure
    This implementation tries to generate multiple candidates in a single request
    """
    # For flash-thinking model, we need special handling
    bon_completion_tokens = 0
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    # Initialize client if needed
    if not hasattr(client, "_is_gemini_initialized"):
        client = genai.Client(api_key=gemini_api_key)
        client._is_gemini_initialized = True
        client._model_name = model
    
    model_name = getattr(client, "_model_name", model)
    
    # Format the prompt - simpler approach without the "INSERT_INPUT_HERE"
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"{system_prompt}\n\n{initial_query}")]
    )
    
    contents = [user_content]
    
    # Configure generation parameters with candidate_count to get multiple responses in one request
    generate_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        candidate_count=min(n, 8),  # Request multiple candidates like OpenAI's n parameter
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="OFF",  # Off
            ),
        ],
        response_mime_type="text/plain",
    )
    
    # Make the request
    logger.info(f"Generating completion using gemini-flash-thinking model with {generate_config.candidate_count} candidates")
    try:
        # Add rate limiting for API calls
        def rate_limited_api_call(func, *args, **kwargs):
            max_retries = 5
            base_delay = 3  # seconds
            
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        # Rate limit hit - exponential backoff
                        delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        # Different error - re-raise
                        raise
            
            # If we've exhausted all retries
            raise Exception(f"Failed after {max_retries} retries due to rate limiting")
        
        # Make a non-streaming request to get multiple candidates at once
        response = rate_limited_api_call(
            client.models.generate_content,
            model=model_name,
            contents=contents,
            config=generate_config
        )
        
        logger.info(f"Response type: {type(response)}")
        
        # Extract all candidates
        completions = []
        
        # Check if we got candidates in the response
        if hasattr(response, 'candidates') and response.candidates:
            logger.info(f"Received {len(response.candidates)} candidates from Gemini")
            
            for i, candidate in enumerate(response.candidates):
                text = None
                
                # Extract text from parts
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text = part.text
                            break
                
                if text:
                    completions.append(text)
                    logger.info(f"Extracted candidate {i+1} with {len(text)} chars")
                    
        # If we have a direct text response and no candidates, use that
        elif hasattr(response, 'text') and response.text:
            completions.append(response.text)
            logger.info(f"Extracted single response with {len(response.text)} chars")
        
        # If no candidates were found, log that
        if not completions:
            logger.warning("No valid candidates found in response")
            
        # Rating the candidates or just selecting the best one 
        if len(completions) <= 1:
            # If there's only one or zero completions, return it
            return completions[0] if completions else "No valid response"
        else:
            # Rate the completions to find the best one
            logger.info(f"Rating {len(completions)} candidates from Gemini")
            
            # Prepare to collect data for saving
            test_data = {
                "model": model_name,
                "system_prompt": system_prompt,
                "query": initial_query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidates": [],
                "final_selection": None
            }
            
            # Use the same rating function from the standard implementation
            ratings = []
            candidate_details = []
            
            for i, completion in enumerate(completions):
                # Track this candidate
                candidate = {
                    "id": i+1,
                    "text": completion,
                    "rating": None,
                    "rating_text": None,
                    "final_selection": False
                }
                
                try:
                    # Create a specialized prompt to get detailed rating
                    rating_content = types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"""Please evaluate the quality of the following response.
First, rate it on a scale from 0 to 10, where 0 is poor and 10 is excellent.
Then provide a brief explanation of your rating (1-2 sentences).

Format your answer as:
RATING: [number]
EXPLANATION: [your explanation]

Here is the response to evaluate:

{completion}""")]
                    )
                    
                    # Rate with exponential backoff for rate limiting
                    rating_response = rate_limited_api_call(
                        client.models.generate_content,
                        model=model_name,
                        contents=[rating_content],
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=2000
                        )
                    )
                    
                    # Extract rating and explanation
                    rating = 5.0  # Default rating
                    rating_text = ""
                    
                    if hasattr(rating_response, 'text') and rating_response.text:
                        response_text = rating_response.text
                        
                        # Try to parse the rating
                        try:
                            # Look for the RATING: pattern
                            if "RATING:" in response_text:
                                rating_part = response_text.split("RATING:")[1].split("\n")[0].strip()
                                rating = float(rating_part)
                            else:
                                # Try to extract any number
                                import re
                                number_match = re.search(r'(\d+(\.\d+)?)', response_text)
                                if number_match:
                                    rating = float(number_match.group(1))
                            
                            # Extract explanation if available
                            if "EXPLANATION:" in response_text:
                                rating_text = response_text.split("EXPLANATION:")[1].strip()
                            else:
                                # Just use the whole response if we can't find the pattern
                                rating_text = response_text
                        except ValueError:
                            rating_text = response_text
                    
                    ratings.append(rating)
                    
                    # Update the candidate details
                    candidate["rating"] = rating
                    candidate["rating_text"] = rating_text
                    
                    logger.info(f"Candidate {i+1} rated: {rating} - {rating_text[:50]}...")
                except Exception as e:
                    logger.error(f"Error rating candidate {i+1}: {str(e)}")
                    ratings.append(5.0)  # Default rating on error
                    candidate["rating"] = 5.0
                    candidate["rating_text"] = f"Error during rating: {str(e)}"
                
                # Add to our list of candidates
                candidate_details.append(candidate)
                test_data["candidates"].append(candidate)
                
                # Add delay between ratings to avoid rate limits
                if i < len(completions) - 1:
                    time.sleep(1)
            
            # Determine the best candidate
            best_response = ""
            if ratings:
                best_index = ratings.index(max(ratings))
                logger.info(f"Selected best candidate {best_index+1} with rating {ratings[best_index]}")
                best_response = completions[best_index]
                
                # Mark the selected candidate
                if best_index < len(candidate_details):
                    candidate_details[best_index]["final_selection"] = True
                    test_data["candidates"][best_index]["final_selection"] = True
                    test_data["final_selection"] = {
                        "index": best_index,
                        "id": best_index + 1,
                        "rating": ratings[best_index],
                        "text": completions[best_index]
                    }
            else:
                best_response = completions[0]  # Fallback if ratings failed
                if candidate_details:
                    candidate_details[0]["final_selection"] = True
                    test_data["candidates"][0]["final_selection"] = True
                    test_data["final_selection"] = {
                        "index": 0,
                        "id": 1,
                        "rating": 5.0, 
                        "text": completions[0]
                    }
            
            # Save the test results to file
            test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_bon_results(test_data, model_name, test_id)
            
            return best_response
            
    except Exception as e:
        logger.error(f"Error in flash-thinking model: {str(e)}")
        # Fall back to standard approach 
        return _best_of_n_sampling_gemini_standard(system_prompt, initial_query, client, model, n)

def _best_of_n_sampling_gemini_standard(system_prompt: str, initial_query: str, client, model: str, n: int = 2) -> str:
    """
    Best-of-N sampling implementation for Gemini API
    """
    bon_completion_tokens = 0
    
    # Check if we already have a Gemini client
    if not hasattr(client, "_is_gemini_initialized"):
        # If not, create a new one
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models")
        
        # Create a new client
        client = genai.Client(api_key=gemini_api_key)
        client._is_gemini_initialized = True
        client._model_name = model
    
    # Get the model from the client
    model_name = getattr(client, "_model_name", model)
    
    # Combine system prompt and query as per the example
    combined_prompt = f"{system_prompt}\n\n{initial_query}"
    
    # Generate multiple samples - limit to 2 for Gemini to avoid rate limits
    num_requests = min(n, 2)  # Only generate 2 completions max to avoid rate limits
    completions = []
    
    # Add rate limiting function
    def rate_limited_api_call(func, *args, **kwargs):
        max_retries = 5
        base_delay = 3  # seconds
        
        for retry in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Rate limit hit - exponential backoff
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry+1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    # Different error - re-raise
                    raise
        
        # If we've exhausted all retries
        raise Exception(f"Failed after {max_retries} retries due to rate limiting")
    
    for i in range(num_requests):
        try:
            # For each request, create a content object
            if i == 0:
                # For the first request, use the standard prompt
                user_content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=combined_prompt)]
                )
                contents = [user_content]
            else:
                # For subsequent requests, add a variation message to get diversity
                variation_prompt = f"{combined_prompt}\n\nProvide a unique perspective (variation {i+1})."
                variation_content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=variation_prompt)]
                )
                contents = [variation_content]
            
            # Configure generation parameters
            generate_config = types.GenerateContentConfig(
                temperature=0.7 + (i * 0.05),  # Vary temperature slightly (0.7 to 1.05)
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
                response_mime_type="text/plain",
            )
            
            # Make the API call using the SDK syntax from the example, with rate limiting
            response = rate_limited_api_call(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=generate_config
            )
            
            # Debug the response structure
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response attributes: {dir(response)}")
            # Try to identify how to access the text
            for attr in ['text', 'parts', 'candidates', 'content', 'result']:
                if hasattr(response, attr):
                    logger.info(f"Found attribute: {attr} = {getattr(response, attr)}")
            
            # Extract the completion
            try:
                if hasattr(response, 'text') and response.text is not None:
                    completions.append(response.text)
                    logger.info(f"Generated completion {i+1}/{num_requests} using Gemini model")
                # Try alternative response structures
                elif hasattr(response, 'parts') and response.parts:
                    # Extract text from parts
                    text_parts = []
                    for part in response.parts:
                        if isinstance(part, dict) and 'text' in part and part['text']:
                            text_parts.append(part['text'])
                    
                    if text_parts:
                        completion_text = ''.join(text_parts)
                        completions.append(completion_text)
                        logger.info(f"Generated completion {i+1}/{num_requests} using Gemini model (from parts)")
                    else:
                        logger.warning(f"No text found in parts for request {i+1}")
                # Check if it's a more direct structure with candidates
                elif hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    completions.append(part.text)
                                    logger.info(f"Generated completion {i+1}/{num_requests} using Gemini model (from candidates)")
                                    break
                else:
                    # Try to extract any string representation as a last resort
                    try:
                        completion_text = str(response)
                        completions.append(completion_text)
                        logger.warning(f"Falling back to string representation for request {i+1}")
                    except:
                        logger.warning(f"Completely unexpected response structure from Gemini API for request {i+1}")
            except Exception as e:
                logger.error(f"Error extracting text from Gemini response: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating Gemini completion {i+1}: {str(e)}")
    
    # If we couldn't generate any completions, handle the error case
    if not completions:
        logger.error("No completions generated with Gemini model")
        return "Error: No completions could be generated."
        
    logger.info(f"Generated {len(completions)} completions with standard Gemini model")
    
    # Prepare to collect data for saving
    test_data = {
        "model": model_name,
        "system_prompt": system_prompt,
        "query": initial_query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidates": [],
        "final_selection": None
    }
    
    # If we only have one completion, just return it without rating but still save the result
    if len(completions) <= 1:
        # Save the single completion as our result
        if completions:
            test_data["candidates"].append({
                "id": 1,
                "text": completions[0],
                "rating": None,
                "rating_text": "Single completion - no rating needed",
                "final_selection": True
            })
            test_data["final_selection"] = {
                "index": 0,
                "id": 1,
                "rating": None,
                "text": completions[0]
            }
            # Save the test results to file
            test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_bon_results(test_data, model_name, test_id)
        
        return completions[0] if completions else ""
        
    # Rate the completions (only if we have multiple)
    ratings = []
    candidate_details = []
    
    # Add a delay before ratings to help avoid rate limiting
    time.sleep(1)
    
    for i, completion in enumerate(completions):
        # Track this candidate
        candidate = {
            "id": i+1,
            "text": completion,
            "rating": None,
            "rating_text": None,
            "final_selection": False
        }
        try:
            # Create a specialized prompt to get detailed rating
            rating_prompt = f"""Please evaluate the quality of the following response.
First, rate it on a scale from 0 to 10, where 0 is poor and 10 is excellent.
Then provide a brief explanation of your rating (1-2 sentences).

Format your answer as:
RATING: [number]
EXPLANATION: [your explanation]

Here is the response to evaluate:

{completion}"""
            
            # Create a content object for rating
            rating_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=rating_prompt)]
            )
            
            # Configure generation parameters for rating
            rating_config = types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for more deterministic rating
                max_output_tokens=200,  # Need space for explanation
                response_mime_type="text/plain",
            )
            
            # Make the API call using the SDK syntax from the example, with rate limiting
            rating_response = rate_limited_api_call(
                client.models.generate_content,
                model=model_name,
                contents=[rating_content],
                config=rating_config
            )
            
            # Debug the rating response structure
            logger.info(f"Rating response type: {type(rating_response)}")
            logger.info(f"Rating response attributes: {dir(rating_response)}")
            # Try to identify how to access the rating text
            for attr in ['text', 'parts', 'candidates', 'content', 'result']:
                if hasattr(rating_response, attr):
                    logger.info(f"Found rating attribute: {attr} = {getattr(rating_response, attr)}")
            
            # Extract rating and explanation
            rating = 5.0  # Default rating
            rating_text = ""
            
            if hasattr(rating_response, 'text') and rating_response.text is not None:
                response_text = rating_response.text
                
                # Try to parse the rating
                try:
                    # Look for the RATING: pattern
                    if "RATING:" in response_text:
                        rating_part = response_text.split("RATING:")[1].split("\n")[0].strip()
                        rating = float(rating_part)
                    else:
                        # Try to extract any number
                        import re
                        number_match = re.search(r'(\d+(\.\d+)?)', response_text)
                        if number_match:
                            rating = float(number_match.group(1))
                    
                    # Extract explanation if available
                    if "EXPLANATION:" in response_text:
                        rating_text = response_text.split("EXPLANATION:")[1].strip()
                    else:
                        # Just use the whole response if we can't find the pattern
                        rating_text = response_text
                except ValueError:
                    rating_text = response_text
                    
            ratings.append(rating)
            
            # Update the candidate details
            candidate["rating"] = rating
            candidate["rating_text"] = rating_text
            
            logger.info(f"Candidate {i+1} rated: {rating} - {rating_text[:50]}...")
                
        except Exception as e:
            logger.error(f"Error rating Gemini completion {i+1}: {str(e)}")
            # If rating fails, assign a default neutral rating
            ratings.append(5.0)
            candidate["rating"] = 5.0
            candidate["rating_text"] = f"Error during rating: {str(e)}"
        
        # Add to our list of candidates
        candidate_details.append(candidate)
        test_data["candidates"].append(candidate)
        
        # Add delay between ratings to avoid rate limits
        if i < len(completions) - 1:
            time.sleep(1)
    
    # Determine the best candidate
    best_response = ""
    if ratings:
        best_index = ratings.index(max(ratings))
        logger.info(f"Selected best candidate {best_index+1} with rating {ratings[best_index]}")
        best_response = completions[best_index]
        
        # Mark the selected candidate
        if best_index < len(candidate_details):
            candidate_details[best_index]["final_selection"] = True
            test_data["candidates"][best_index]["final_selection"] = True
            test_data["final_selection"] = {
                "index": best_index,
                "id": best_index + 1,
                "rating": ratings[best_index],
                "text": completions[best_index]
            }
    else:
        best_response = completions[0]  # Fallback if ratings failed
        if candidate_details:
            candidate_details[0]["final_selection"] = True
            test_data["candidates"][0]["final_selection"] = True
            test_data["final_selection"] = {
                "index": 0,
                "id": 1,
                "rating": 5.0, 
                "text": completions[0]
            }
    
    # Save the test results to file
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_bon_results(test_data, model_name, test_id)
    
    return best_response