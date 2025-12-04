import os
from llama_cpp import Llama
import json
import re
import argparse
import ast


def generate_response(_model: Llama, _messages: list) -> str:
    try:
        _output = _model.create_chat_completion(
            messages=_messages,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            max_tokens=16384,
            temperature=0.75,
        )["choices"][0]["message"]["content"]
        return _output.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


def extract_json(text):
    """
    Extracts a valid JSON object from the model output.
    Fixes cases where the model outputs extra text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)  # Extract JSON block
    if match:
        json_str = match.group(0)
        try:
            parsed_json = json.loads(json_str)  # Parse JSON correctly
            if isinstance(parsed_json, dict) and "persona" in parsed_json:
                # Extract only the 'persona' text
                return parsed_json["persona"].strip()
        except json.JSONDecodeError:
            print("Warning: Generated JSON is invalid, skipping...")
    return None  # Return None if extraction fails


gender_random_list = [
    ('teen (15-19)', 'Farmer', 'Extraversion (E)'),
    ('teen (15-19)', 'Software Engineer', 'Introversion (I)'),
    ('teen (15-19)', 'Investment Analyst', 'Feeling (F)'),
    ('teen (15-19)', 'Primary School Teacher', 'Extraversion (E)'),
    ('teen (15-19)', 'Doctor', 'Introversion (I)'),
    ('adult (20-45)', 'Actor', 'Extraversion (E)'),
    ('adult (20-45)', 'Farmer', 'Introversion (I)'),
    ('adult (20-45)', 'Software Engineer', 'Feeling (F)'),
    ('adult (20-45)', 'Investment Analyst', 'Extraversion (E)'),
    ('adult (20-45)', 'Primary School Teacher', 'Introversion (I)'),
    ('middle-age (45-65)', 'Doctor', 'Extraversion (E)'),
    ('middle-age (45-65)', 'Actor', 'Introversion (I)'),
    ('middle-age (45-65)', 'Farmer', 'Feeling (F)'),
    ('middle-age (45-65)', 'Software Engineer', 'Extraversion (E)'),
    ('middle-age (45-65)', 'Investment Analyst', 'Introversion (I)'),
    ('elderly (65 up)', 'Primary School Teacher', 'Extraversion (E)'),
    ('elderly (65 up)', 'Doctor', 'Introversion (I)'),
    ('elderly (65 up)', 'Actor', 'Feeling (F)'),
    ('elderly (65 up)', 'Farmer', 'Extraversion (E)'),
    ('elderly (65 up)', 'Software Engineer', 'Introversion (I)'),
]
age_random_list = [
    ('male', 'Farmer', 'Extraversion (E)'),
    ('male', 'Farmer', 'Introversion (I)'),
    ('male', 'Software Engineer', 'Feeling (F)'),
    ('male', 'Software Engineer', 'Extraversion (E)'),
    ('male', 'Investment Analyst', 'Introversion (I)'),
    ('male', 'Investment Analyst', 'Feeling (F)'),
    ('male', 'Primary School Teacher', 'Extraversion (E)'),
    ('male', 'Primary School Teacher', 'Introversion (I)'),
    ('male', 'Doctor', 'Extraversion (E)'),
    ('male', 'Actor', 'Introversion (I)'),
    ('female', 'Farmer', 'Extraversion (E)'),
    ('female', 'Farmer', 'Introversion (I)'),
    ('female', 'Software Engineer', 'Feeling (F)'),
    ('female', 'Software Engineer', 'Extraversion (E)'),
    ('female', 'Investment Analyst', 'Introversion (I)'),
    ('female', 'Investment Analyst', 'Feeling (F)'),
    ('female', 'Primary School Teacher', 'Extraversion (E)'),
    ('female', 'Primary School Teacher', 'Introversion (I)'),
    ('female', 'Doctor', 'Extraversion (E)'),
    ('female', 'Actor', 'Introversion (I)'),
]
age_list = ["teen(15-19)", "adult(20-45)",
            "middle-age(45-65)", "elderly(65 up)"]
gender_list = ["male", "female"]


def main():
    os.makedirs('./data/age', exist_ok=True)
    os.makedirs('./data/gender', exist_ok=True)
    
    myModel = Llama(
        "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        verbose=False,
        n_gpu_layers=-1,
        n_ctx=16384,
    )
    prompts = []
    system_prompt = "You are a helpful assistant that is proficient in generating personas."

    for age in age_list:
        for i in range(20):
            user_prompt = f"""Create a detailed and realistic persona for a user simulator based on the following criteria:

                         - **Gender**: {age_random_list[i][0]} 
                         - **Age**: {age}
                         - **Occupation**: {age_random_list[i][1]}  according to the International Standard Industrial Classification (ISIC)
                         - **Name**: Generate according to the gender (different names every time).  
                         - **Personality Traits**: {age_random_list[i][2]}  according to the Myers–Briggs Type Indicator (MBTI).

                         ### **Objective:**  
                         The goal is to generate well-rounded personas that explicitly reflect the provided gender, age, and occupation. These personas should illustrate how each individual engages with their surroundings, expresses themselves, and navigates social and professional interactions.  
                         Directly generate a unique persona, make sure you specify the age, the gender, and the occupation.
                         
                         ### **Output Format (Strict JSON)**  
                         Respond **ONLY** with a valid JSON object, following this exact format:  
                         ```json
                         {{
                         "persona": "You're [Name], a [Age]-year-old [gender] [Occupation] who [personality-driven description]. [Other descriptions]"
                         }}
                         ```

                         ### **Sample output:**  
                         {{
                              "persona": "You're Emily Thompson, a 28-year-old female marketing specialist who thrives in dynamic environments. You love brainstorming creative campaigns, networking at industry events, and sharing innovative ideas with colleagues. Outside of work, you enjoy hiking in the mountains, playing guitar at open mic nights, and engaging in social activities that keep your energy levels high."
                         }}

                         Ensure that:  
                         - The JSON output is **well-formed and properly formatted**.  
                         - The persona is natural and unique each time.  
                         - Do not include additional explanations or formatting outside of the JSON output.
                         - You have to come up with different names everytime so be creative on names.
                         - The age should be within the age range.
                         """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            output_text = generate_response(myModel, messages)
            persona_text = extract_json(output_text)
            if persona_text:
                prompts.append({"id": (i + 1), "persona": persona_text})
                print(persona_text)
            else:
                print(f"Invalid JSON response at index {i}, skipping...")

        age = age.split('(')[0]
        with open(f"./data/age/{age}.json", "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=4, ensure_ascii=False)

    for gender in gender_list:
        for i in range(20):
            user_prompt = f"""Create a detailed and realistic persona for a user simulator based on the following criteria:

                         - **Gender**: {gender} 
                         - **Age**: {gender_random_list[i][0]}
                         - **Occupation**: {gender_random_list[i][1]}  according to the International Standard Industrial Classification (ISIC)
                         - **Name**: Generate according to the gender (different names every time).  
                         - **Personality Traits**: {gender_random_list[i][2]}  according to the Myers–Briggs Type Indicator (MBTI).

                         ### **Objective:**  
                         The goal is to generate well-rounded personas that explicitly reflect the provided gender, age, and occupation. These personas should illustrate how each individual engages with their surroundings, expresses themselves, and navigates social and professional interactions.  
                         Directly generate a unique persona, make sure you specify the age, the gender, and the occupation.
                         
                         ### **Output Format (Strict JSON)**  
                         Respond **ONLY** with a valid JSON object, following this exact format:  
                         ```json
                         {{
                         "persona": "You're [Name], a [Age]-year-old [gender] [Occupation] who [personality-driven description]. [Other descriptions]"
                         }}
                         ```

                         ### **Sample output:**  
                         {{
                              "persona": "You're Emily Thompson, a 28-year-old female marketing specialist who thrives in dynamic environments. You love brainstorming creative campaigns, networking at industry events, and sharing innovative ideas with colleagues. Outside of work, you enjoy hiking in the mountains, playing guitar at open mic nights, and engaging in social activities that keep your energy levels high."
                         }}

                         Ensure that:  
                         - The JSON output is **well-formed and properly formatted**.  
                         - The persona is natural and unique each time.  
                         - Do not include additional explanations or formatting outside of the JSON output.
                         - You have to come up with different names everytime so be creative on names.
                         - The age should be within the age range.
                         """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            output_text = generate_response(myModel, messages)
            persona_text = extract_json(output_text)
            if persona_text:
                prompts.append({"id": (i + 1), "persona": persona_text})
                print(persona_text)
            else:
                print(f"Invalid JSON response at index {i}, skipping...")

        with open(f"./data/gender/{gender}.json", "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
