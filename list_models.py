import google.generativeai as genai

API_KEY = 'AIzaSyCXP3gztyXK0sgerGdi6oRuZGUh4TvIvi4'  # Replace with your key

genai.configure(api_key=API_KEY)

print("Available Gemini models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  ✓ {m.name}")