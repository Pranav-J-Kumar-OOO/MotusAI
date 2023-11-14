from flask import Flask, render_template, request
import openai
import os
import time


app = Flask(__name__)
f = 1
# Set up OpenAI API credentials
openai.api_key = "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"


# Define the default route to return the index.html file
@app.route("/")
def index():
    return render_template("index.html")

# Define the /api route to handle POST requests
@app.route("/api", methods=["POST"])
def api():
    #os.startfile("videotester.py")
    print("FAQ")
    #time.sleep(6)
    # Get the message from the POST request

    message = request.json.get("message")
    # Send the message to OpenAI's API and receive the response
    
    
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": message}
    ]
    )
    if completion.choices[0].message!=None:
        return completion.choices[0].message

    else :
        return 'Failed to Generate response!'
    

if __name__=='__main__':
    app.run()

