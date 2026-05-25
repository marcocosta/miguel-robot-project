from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.2",
    input="Say hello to Chief Engineer Marquinho and Systems Engineer Marco in one short sentence."
)

print(response.output_text)
