import os
from flask import Flask, render_template, request, jsonify
from google import generativeai as genai
from dotenv import load_dotenv

# Load environment variables (especially API key)
load_dotenv()

app = Flask(__name__)

# Configure the Google Generative AI client
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash') # Use a suitable model
except Exception as e:
    print(f"Error configuring GenAI: {e}")
    # Handle the error appropriately, maybe exit or use a fallback
    model = None

# System instruction for the AI model
SYSTEM_INSTRUCTION = """You are an AI Debt Repayment Planner chatbot. Your goal is to help users create a plan to pay off their debts.
You will NEVER answer any questions about anything other than debt repayment.
You use plain text format never markdown.
You should:
1. Ask the user for their debt details: name/type of debt, total amount owed, interest rate (APR), and minimum monthly payment for each debt.
2. Ask the user for their total monthly budget allocated for debt repayment (this should be at least the sum of all minimum payments).
3. Based on the provided information, explain and calculate repayment plans using:
    a. The Debt Snowball method (paying off smallest debts first).
    b. The Debt Avalanche method (paying off highest interest rate debts first).
    c. A personalized suggestion, considering their budget and potentially offering variations or hybrid approaches if applicable.
4. Present the plans clearly, showing the order of payoff, estimated time to become debt-free, and total interest paid for each method.
5. Keep the interaction conversational and encouraging.
6. If the user's budget is less than the sum of minimum payments, point this out and explain that they need to allocate at least that much.
"""

@app.route('/')
def index():
    """Renders the main chat page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages and interacts with the AI."""
    if not model:
        return jsonify({"error": "AI model not configured."}), 500

    user_message = request.json.get('message')
    # In a real app, you'd manage conversation history
    # For simplicity here, we'll send the system instruction and user message each time
    # A more robust solution would use model.start_chat()

    prompt_content = [
        {"role": "user", "parts": [SYSTEM_INSTRUCTION]},
        {"role": "model", "parts": ["Okay, I understand. I'm ready to help you with your debt repayment plan. Please tell me about your debts (name, amount, interest rate, minimum payment) and your monthly budget for debt repayment."]}, # Start the conversation
        {"role": "user", "parts": [user_message]}
    ]

    try:
        # Note: For a stateful conversation, use model.start_chat(history=...)
        # history = [{"role": "user", "parts": [SYSTEM_INSTRUCTION]}, {"role": "model", "parts": [...]}]
        # chat_session = model.start_chat(history=history)
        # response = chat_session.send_message(user_message)
        # For this simple example, we continue sending the context each time.
        response = model.generate_content(prompt_content)
        ai_message = response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        ai_message = "Sorry, I encountered an error trying to process your request."

    return jsonify({"reply": ai_message})

if __name__ == '__main__':
    # Make sure to create a 'templates' folder in the same directory as app.py
    # And create an 'index.html' file inside 'templates'
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Always overwrite index.html with the latest version during development run
    # In production, this file would typically be static.
    with open('templates/index.html', 'w') as f:
        # Updated HTML with ChatGPT-like UI
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Debt Planner</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom scrollbar for chatbox */
        #chatbox::-webkit-scrollbar {
            width: 6px;
        }
        #chatbox::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        #chatbox::-webkit-scrollbar-thumb {
            background: #c1c1c1; /* Lighter scrollbar thumb */
            border-radius: 10px;
        }
        #chatbox::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        /* Ensure input doesn't overlap button */
        #userInput {
            padding-right: 4rem; /* Space for the button */
        }
    </style>
</head>
<body class="bg-gray-100 h-screen flex flex-col font-sans">

    <main class="flex-grow container mx-auto p-4 flex flex-col max-w-3xl w-full">
        <!-- Chat messages area -->
        <div id="chatbox" class="flex-grow overflow-y-auto mb-4 space-y-6 p-4">
            <!-- Initial AI Message -->
             <div class="flex justify-start group">
                <div class="bg-white text-gray-800 rounded-lg px-4 py-3 max-w-xl shadow-sm border border-gray-200">
                    <p class="font-semibold text-sm mb-1 text-gray-700">AI Debt Planner:</p>
                    <p class="text-sm">Hello! I'm here to help you create a debt repayment plan. Please tell me about your debts (name, amount owed, APR %, minimum monthly payment) and your total monthly budget for debt repayment.</p>
                </div>
            </div>
            <!-- Chat messages will appear here -->
        </div>

        <!-- Composer area -->
        <div class="mt-auto px-4 pb-4 pt-2 bg-gray-100 sticky bottom-0">
            <div class="relative flex items-center bg-white rounded-2xl shadow-lg border border-gray-300 overflow-hidden">
                <input type="text" id="userInput" class="flex-grow border-none focus:ring-0 bg-transparent px-5 py-3 text-gray-800 placeholder-gray-500 text-sm" placeholder="Enter your debt details or message...">
                <button id="sendButton" class="absolute right-2 bottom-2 bg-black text-white rounded-lg w-8 h-8 flex items-center justify-center hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-black disabled:bg-gray-300" disabled>
                    <!-- SVG Arrow Icon -->
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path fill-rule="evenodd" clip-rule="evenodd" d="M12.7071 4.29289C13.0976 4.68342 13.0976 5.31658 12.7071 5.70711L7.41421 11H20C20.5523 11 21 11.4477 21 12C21 12.5523 20.5523 13 20 13H7.41421L12.7071 18.2929C13.0976 18.6834 13.0976 19.3166 12.7071 19.7071C12.3166 20.0976 11.6834 20.0976 11.2929 19.7071L4.29289 12.7071C3.90237 12.3166 3.90237 11.6834 4.29289 11.2929L11.2929 4.29289C11.6834 3.90237 12.3166 3.90237 12.7071 4.29289Z" transform="rotate(180 12 12)" fill="currentColor"/>
                    </svg>
                </button>
            </div>
             <div class="text-center text-xs text-gray-500 mt-2">
                AI Debt Planner can make mistakes. Consider checking important information.
            </div>
        </div>
    </main>

    <script>
        const chatbox = document.getElementById('chatbox');
        const userInput = document.getElementById('userInput');
        const sendButton = document.getElementById('sendButton');

        function addMessage(sender, message) {
            const messageContainer = document.createElement('div');
            const messageBubble = document.createElement('div');
            messageBubble.classList.add('rounded-lg', 'px-4', 'py-3', 'max-w-xl', 'shadow-sm', 'border');

            const senderP = document.createElement('p');
            senderP.classList.add('font-semibold', 'text-sm', 'mb-1');

            const messageP = document.createElement('p');
            messageP.classList.add('text-sm', 'whitespace-pre-wrap'); // Use pre-wrap to respect newlines
            // Basic sanitization to prevent HTML injection - only display text content
            messageP.textContent = message;

            messageBubble.appendChild(senderP);
            messageBubble.appendChild(messageP);

            if (sender === 'You') {
                messageContainer.classList.add('flex', 'justify-end', 'group'); // Align user messages to the right
                messageBubble.classList.add('bg-blue-600', 'text-white', 'border-blue-700');
                senderP.textContent = sender + ':';
                senderP.classList.add('text-blue-100');
            } else { // AI
                messageContainer.classList.add('flex', 'justify-start', 'group'); // Align AI messages to the left
                messageBubble.classList.add('bg-white', 'text-gray-800', 'border-gray-200');
                senderP.textContent = 'AI Debt Planner:';
                senderP.classList.add('text-gray-700');
            }

            messageContainer.appendChild(messageBubble);
            chatbox.appendChild(messageContainer);
            chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage('You', message);
            userInput.value = ''; // Clear input
            userInput.disabled = true; // Disable input while waiting
            sendButton.disabled = true; // Disable send button
            sendButton.classList.add('animate-pulse'); // Optional: Add thinking indicator

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ reply: 'Unknown server error' })); // Try to parse error
                    throw new Error(`HTTP error! status: ${response.status} - ${errorData.reply || errorData.error}`);
                }

                const data = await response.json();
                addMessage('AI', data.reply);

            } catch (error) {
                console.error('Error sending message:', error);
                addMessage('AI', `Sorry, I encountered an error: ${error.message}`);
            } finally {
                 userInput.disabled = false; // Re-enable input
                 sendButton.disabled = userInput.value.trim() === ''; // Re-enable send button only if input has text
                 sendButton.classList.remove('animate-pulse'); // Remove thinking indicator
                 userInput.focus(); // Keep focus on input
            }
        }

        // Enable/disable send button based on input content
        userInput.addEventListener('input', () => {
            sendButton.disabled = userInput.value.trim() === '';
        });

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) { // Send on Enter
                e.preventDefault(); // Prevent default Enter behavior (like form submission or newline)
                if (!sendButton.disabled) { // Only send if button is enabled
                    sendMessage();
                }
            }
        });

        // Add initial focus to the input field
        userInput.focus();

    </script>
</body>
</html>
""")
    print("Created/Updated templates/index.html with ChatGPT-like UI")

    # It's recommended to use a proper WSGI server like gunicorn or waitress in production
    app.run(debug=True) # Use debug=True for development only