# Semantic-Debugger
NLP-Based Error Resolution Engine using Stack Overflow Data
AI-Powered Debugging Assistant 🤖
A smart tool that helps Python developers debug their code faster by suggesting Stack Overflow solutions for their error messages.

🤔 The Problem
We've all been there. You run your code, and you get a confusing TypeError or a KeyError.

What's next? You copy the error, open a new browser tab, search Google, and then click through five different Stack Overflow links trying to find the one that actually solves your problem.

This breaks your concentration and wastes valuable time.

✨ The Solution
This project is a simple debugging helper that does the hard work for you. Instead of searching, you just give the tool your error message.

It uses Machine Learning and Natural Language Processing (NLP) to "read" the error, understand what it means, and instantly find the most relevant, high-quality solution from a massive dataset of Stack Overflow posts.

In short: You give it an error, it gives you a solution.

🚀 Live Demo & Screenshots
Here is a look at the simple web app (built with Streamlit). You just paste your error message and get a suggested solution!

``

The project also runs as a separate REST API, so other applications or even an IDE plugin could use it.

![WhatsApp Image 2025-10-29 at 23 50 32_d303017f](https://github.com/user-attachments/assets/e6ddac3a-5d0a-4b4c-aaf5-4d2802564b0f)![WhatsApp Image 2025-10-29 at 23 51 18_b9cda824](https://github.com/user-attachments/assets/86c0b8ec-42a3-49f2-b8de-1e594b663af0)


💻 How It Works: The Recipe
You can think of this project as a 5-step recipe:

Get The Ingredients (Data): We started with a huge dataset of Python questions and answers from Stack Overflow (thanks to Kaggle).

Prep The Kitchen (Cleaning): This data was messy. We had to clean up all the text, code snippets, and titles to make them usable for a machine.

Teach The Machine (NLP): We used a technique called [Your Technique, e.g., TF-IDF or BERT Embeddings] to turn all those words and error codes into numbers (vectors) that a machine learning model can understand.

Create The Main Dish (The Model): We trained a [Your Model, e.g., Logistic Regression / Naive Bayes] model to classify the type of error and find the most similar, highest-voted solutions from the dataset. It achieved [Your Metric, e.g., 85% accuracy] on our test data.

Serve It (Deployment): We served the model in two ways:

An interactive Streamlit Web App for humans to use easily.

A Flask/FastAPI REST API for other machines or tools to connect to.

🛠️ Tech Stack
Core ML/Data: Python, Pandas, Scikit-learn, [NLTK/spaCy]

Model: [e.g., Logistic Regression, TensorFlow, PyTorch]

API: Flask

Web App: Streamlit 

Deployment: Docker

🏁 Getting Started
Want to run this on your own machine? Here's how:

Prerequisites:

Python 3.8 or later

You will need to download the dataset from https://www.kaggle.com/datasets/stackoverflow/pythonquestions and place it in the [e.g., /data] folder.

1. Clone the repository:

Bash

git clone (https://github.com/divyanshupathak01/Semantic-Debugger.git)
cd Semantic-Debugger
2. Create and activate a virtual environment:

Bash

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
3. Install the dependencies: (Make sure you have a requirements.txt file in your project!)

Bash

pip install -r requirements.txt
4. Run the applications:

To start the Web App (Streamlit):

Bash

streamlit run app.py
Then open http://localhost:8501 in your browser.

To start the API (Flask/FastAPI):

Bash

# For Flask
python api.py

# For FastAPI/Uvicorn
uvicorn api:app --reload
The API will be running at http://localhost:5000 (or 8000).

🚀 Using the API
Once the API is running, you can send it a POST request with your error message.

Here’s an example using curl:

Bash

curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"error_message": "TypeError: can only concatenate str (not \"int\") to str"}'
Expected Response:

JSON

{
  "solution_title": "TypeError: can only concatenate str (not \"int\") to str in Python",
  "solution_url": "https://stackoverflow.com/questions/123456/...",
  "confidence_score": 0.92
}
📈 Future Improvements
[ ] Integrate a more advanced model (like BERT) for better context understanding.

[ ] Build a simple VS Code extension that uses the API.

[ ] Expand the dataset to include JavaScript and Java errors.

👤 Contact
Find me on:

LinkedIn: https://www.linkedin.com/in/divyanshu-pathak-a05308338?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BGSPLDXehQgetofIQFKi9GA%3D%3D

GitHub: [Your GitHub Profile URL]

Portfolio: [Your Portfolio Website URL]
