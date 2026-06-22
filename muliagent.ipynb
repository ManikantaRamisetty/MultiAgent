from google import genai # we need to generate text so we are using this lib
from google.genai.types import GenerateContentConfig

# req LLM Model , we are using LLM using API key
from google.colab import userdata
# access the API Keyy
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL  = "gemini-2.5-flash"

# make the core function

def run_agent(system_prompt : str, user_prompt :str,temperature=0.7) -> str:
  response = client.models.generate_content(
      model = MODEL,
      contents = [
          {"role" : "user", "parts": [{"text":system_prompt}] },
          {"role" : "user", "parts": [{"text":user_prompt}] },
      ],
      config = GenerateContentConfig(
          temperature = temperature,
          max_output_tokens=2048
      )
  )
  return response.text.strip()

# agent: researcher agent
def researcher(topic:str) -> str: # return string
  system = "You are the research Agent.return the exactly 5 concise bullet-point facts"
  user = f"Research this topic: {topic}"
  result = run_agent(system,user)
  print(result)
  return result


# agent: critical agent find the error
def critic(research:str) -> str: # return string
  system = "You are the critical review reviewer Agent.Identify 2-3 gaps, errors or improvements"
  user = f"Reviewer this critic: {research}"
  result = run_agent(system,user)
  print(result)
  return result

# agent: writer agent
def writer(research:str, critic:str) -> str: # return string
  system = "You are the Professional writer.write the clean , polished 3-sentence summary using research & address the critic"
  user = f"Reviewer this writer: {critic}"
  result = run_agent(system,user)
  print(result)
  return result


# run the code in pipeline
# main pipeline
def run_pipeline(topic:str):
  notes = researcher(topic) # Agent 1
  feedback = critic(notes)  # A2
  summary = writer(notes,feedback)

  return {
      "research": notes,
      "critic": feedback,
      "summary": summary
  }

# ==================== Run ====================
if __name__ == "__main__":
    result = run_pipeline("The future of quantum computing")
    # Optional: display final summary nicely
    print("\n Final Summary:")
    print(result["summary"])
