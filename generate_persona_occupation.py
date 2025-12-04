import os
from typing import List
from llama_cpp import Llama
import json
from pydantic import BaseModel
from tqdm import tqdm
import re
import argparse
import ipdb


def extract_json(text):
    try:
        # Match JSON object or array enclosed by square or curly brackets
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print("Error: No valid JSON structure found.")
            return []
    except json.JSONDecodeError:
        print("Error: Model output is not valid JSON.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def generate_response(model: Llama, messages: str) -> str:
    '''
    This function will inference the model with given messages.
    '''
    output = model.create_chat_completion(
        messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        # This argument is how many tokens the model can generate, you can change it and observe the differences.
        max_tokens=512,
        # This argument is the randomness of the model. 0 means no randomness. You will get the same result with the same input every time. You can try to set it to different values.
        temperature=0.7,
        repeat_penalty=2.0,
    )["choices"][0]["message"]["content"]
    return output


sections = {
    'A': 'Agriculture, forestry and fishing',
    'B': 'Mining and quarrying',
    'C': 'Manufacturing',
    'D': 'Electricity, gas, steam and air conditioning supply',
    'E': 'Water supply; sewerage, waste management and remediation activities',
    'F': 'Construction',
    'G': 'Wholesale and retail trade; repair and selling of motor vehicles and motorcycles',
    'H': 'Transportation and storage',
    'I': 'Accommodation and food service activities',
    'J': 'Information and communication',
    'K': 'Financial and insurance activities',
    'L': 'Real estate activities',
    'M': 'Professional, scientific and technical activities',
    'N': 'Administrative and support service activities',
    'O': 'Public administration and defence; compulsory social security',
    'P': 'Education',
    'Q': 'Human health and social work activities',
    'R': 'Arts, entertainment and recreation',
    'S': 'Other service activities',
    'T': 'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use',
    'U': 'Activities of extraterritorial organizations and bodies',
}

occupations = {
    "A": "Agricultural Scientist",
    "B": "Mining Engineer",
    "C": "Industrial Production Manager",
    "D": "Power Plant Operator",
    "E": "Environmental Engineer",
    "F": "Civil Engineer",
    "G": "Retail Manager",
    "H": "Logistics Coordinator",
    "I": "Hotel Manager",
    "J": "Software Developer",
    "K": "Financial Analyst",
    "L": "Real Estate Agent",
    "M": "Data Scientist",
    "N": "Human Resources Manager",
    "O": "Policy Analyst",
    "P": "University Professor",
    "Q": "Nurse",
    "R": "Graphic Designer",
    "S": "Wedding Planner",
    "T": "Private Chef",
    "U": "Diplomat"
}


prompt = """\
Create a detailed and realistic persona for a user simulator based on the following criteria:

- **Gender**: {gender}
- **Age**: {age}
- **Occupation**: {occupation}, according to the International Standard Industrial Classification (ISIC)
- **Name**: Generate according to the gender (different names every time).  
- **Personality Traits**: {personality}, according to the Myers-Briggs Type Indicator (MBTI).

### **Objective:**  
The goal is to generate well-rounded personas that explicitly reflect the provided gender, age, and occupation. These personas should illustrate how each individual engages with their surroundings, expresses themselves, and navigates social and professional interactions.  
Directly generate a unique persona, make sure you specify the age, the gender, and the occupation.

### **Output Format (Strict JSON)**  
Respond **ONLY** with a valid JSON object, following this exact format:  
```json
{{
    "persona": "You're [Name], a [Age]-year-old male [Occupation] who [personality-driven description]. [Other descriptions]"
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
- The age should be within the age range.\
"""

ages = ["teen(15-19)", "adult(20-45)", "middle-age(45-65)", "elderly(65 up)"]

llm = Llama.from_pretrained(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="*Q8_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=65536,
)

if __name__ == '__main__':
    occupations = [
        "Farmer",
        "Woodcutter",
        "Fisherman",
        "Horticulturist",
        "Software Engineer",
        "Cybersecurity Specialist",
        "Data Scientist",
        "Telecommunications Technician",
        "Investment Analyst",
        "Actuary",
        "Insurance Claims Adjuster",
        "Financial Advisor",
        "Primary School Teacher",
        "University Professor",
        "Vocational Trainer",
        "Special Education Teacher",
        "Doctor",
        "Nurse",
        "Physical Therapist",
        "Psychologist",
        "Actor",
        "Musician",
        "Artist",
        "Writer"
    ]

    settings = [
        ('male', 'teen (15-19)', 'Extraversion (E)'),
        ('male', 'teen (15-19)', 'Introversion (I)'),
        ('male', 'teen (15-19)', 'Feeling (F)'),
        ('male', 'adult (20-45)', 'Extraversion (E)'),
        ('male', 'adult (20-45)', 'Introversion (I)'),
        ('male', 'adult (20-45)', 'Feeling (F)'),
        ('male', 'middle-age (45-65)', 'Extraversion (E)'),
        ('male', 'middle-age (45-65)', 'Introversion (I)'),
        ('male', 'elderly (65 up)', 'Extraversion (E)'),
        ('male', 'elderly (65 up)', 'Introversion (I)'),
        ('female', 'teen (15-19)', 'Extraversion (E)'),
        ('female', 'teen (15-19)', 'Introversion (I)'),
        ('female', 'teen (15-19)', 'Feeling (F)'),
        ('female', 'adult (20-45)', 'Extraversion (E)'),
        ('female', 'adult (20-45)', 'Introversion (I)'),
        ('female', 'adult (20-45)', 'Feeling (F)'),
        ('female', 'middle-age (45-65)', 'Extraversion (E)'),
        ('female', 'middle-age (45-65)', 'Introversion (I)'),
        ('female', 'elderly (65 up)', 'Extraversion (E)'),
        ('female', 'elderly (65 up)', 'Introversion (I)'),
    ]

    os.makedirs('./data/occupation', exist_ok=True)

    for occupation in tqdm(occupations):
        personas = []
        for i, (gender, age, personality) in enumerate(tqdm(settings, desc=occupation)):
            while True:
                output_text = generate_response(
                    llm,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that is proficient in generating personas."
                        },
                        {
                            "role": "user",
                            "content": prompt.format(
                                gender=gender,
                                age=age,
                                occupation=occupation,
                                personality=personality
                            )
                        }
                    ]
                )
                persona_text = extract_json(output_text)
                if persona_text:
                    personas.append({"id": i, "persona": persona_text})
                    print(persona_text)
                    break
                else:
                    print(f"Invalid JSON response at index {i}, retry.")

        with open(f'data/occupation/{occupation.lower().replace(" ", "_")}.json', 'w') as f:
            json.dump(personas, f, indent=4)
