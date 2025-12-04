# From Simulation to Strategy: Automating Personalized Interaction Planning for Conversational Agents

## Environment

- Python 3.11
- Install dependencies from `requirements.txt`

## Persona Generation

Run the following scripts to generate personas:

> **Note:** You will need the `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` model file for persona generation.

```bash
python generate_persona_age_gender.py
python generate_persona_occupation.py
```

## Simulation

- For **Persona Attribute Analysis**, run `simulation_exp1.sh`.  
    Ensure you are hosting `Llama-3.1-8B-Instruct` and update the endpoint URL in the code as needed.

- For **Occupation-Based Strategy for SalesAgent**, run `simulation_exp2_no_strategy.sh` and `simulation_exp2_strategy.sh`.  
    You will need to host `Qwen3-8B` and `mistral-7b-instruct-v0.3`, and modify the endpoint URLs accordingly.

## Analysis

Refer to the following notebooks for analysis:

- `age_gender_analysis.ipynb`
- `occupation_analysis.ipynb`