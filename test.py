import openai
import json
import instructor
import re

# Variable globale pour contrôler les messages de debug
DEBUG = False

client = openai.OpenAI(
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # can be anything
    base_url = "https://caronboulme.fr/llm/v1"
)
client = instructor.patch(client=client)
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    messages = [{"role": "user", "content": "Quel temps fait-il à San Francisco, Tokyo et Paris ?"}]
    # follow this page for OpenAI function calling instruction
    # https://platform.openai.com/docs/guides/function-calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="functionary", # the model name doesn't really matter in this case.
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # DEBUG: Print response and tool calls
    if DEBUG:
        print("DEBUG: Response message:", response_message)
        print("DEBUG: Tool calls:", tool_calls)
    
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        # Step 4: send the info for each function call and function response to the model
        for i, tool_call in enumerate(tool_calls):
            if DEBUG:
                print(f"DEBUG: Tool call {i}:")
                print(f"  - ID: {tool_call.id}")
                print(f"  - Function name: '{tool_call.function.name}'")
                print(f"  - Function args: '{tool_call.function.arguments}'")
            
            # Parse multiple function calls from malformed function name
            raw_function_data = tool_call.function.name
            
            if DEBUG:
                print(f"DEBUG: Raw function data: '{raw_function_data}'")
            
            # Extract all JSON objects that look like function arguments
            json_pattern = r'\{[^{}]*"location"[^{}]*\}'
            json_matches = re.findall(json_pattern, raw_function_data)
            
            if DEBUG:
                print(f"DEBUG: Found {len(json_matches)} JSON matches: {json_matches}")
            
            # Process each found JSON as a separate function call
            for j, json_str in enumerate(json_matches):
                if DEBUG:
                    print(f"DEBUG: Processing function call {j}: JSON='{json_str}'")
                
                try:
                    # Clean the JSON string by removing any trailing tokens
                    clean_json = re.sub(r'<\|.*$', '', json_str).strip()
                    function_args = json.loads(clean_json)
                    function_name = "get_current_weather"  # We know this is the function being called
                    
                    if DEBUG:
                        print(f"DEBUG: Extracted function name: '{function_name}'")
                        print(f"DEBUG: Extracted function args: {function_args}")
                    
                    if function_name not in available_functions:
                        if DEBUG:
                            print(f"ERROR: Function '{function_name}' not found")
                        continue
                    
                    function_to_call = available_functions[function_name]
                except json.JSONDecodeError as e:
                    if DEBUG:
                        print(f"ERROR: Failed to parse JSON '{json_str}': {e}")
                    continue
                
                # call function to get the result
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                
                messages.append(
                    {
                        "tool_call_id": f"{tool_call.id}_{j}",
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
        for message in messages:
            # Function call responses
            if message["role"] == "function" and "name" in message:
                message["name"] = f"functions.{message['name']}"
        second_response = client.chat.completions.create(
            model="functionary",
            messages=messages,
            stream=True
        )  # get a new response from the model where it can see the function response
        return second_response

def stream_response():
    response_stream = run_conversation()
    
    print("", end='', flush=True)  # Initialize output
    for chunk in response_stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print()  # New line at the end

stream_response()
