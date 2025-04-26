import os
import random
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from google import generativeai as genai
from dotenv import load_dotenv
import json # Added for formatting debt data for the prompt

# Load environment variables (especially API key)
load_dotenv()

app = Flask(__name__)

# Configure the Google Generative AI client
try:
    # Use the API key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
    # Select the appropriate model (using the one from the chat example)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
except Exception as e:
    print(f"Error configuring GenAI: {e}")
    # Handle the error appropriately, maybe exit or use a fallback
    model = None

# System instruction for the AI model (from the chat example)
SYSTEM_INSTRUCTION = """
**Your Role:** You are an AI financial advisor specializing *exclusively* in debt repayment planning.

**Your Core Task:**
1.  Analyze user-provided debt details (like name, principal balance, APR interest rate, minimum monthly payment). This data might be provided via a form or directly in the chat.
2.  Analyze the user's stated total monthly budget allocated specifically for debt repayment.
3.  Based on the debt data and budget, discuss and explain potential debt repayment strategies (such as the Debt Snowball or Debt Avalanche methods).
4.  Answer clarifying questions *strictly related* to the provided debt data, budget, and repayment strategies.
5.  If debt data or budget is missing, politely request it.

**Strict Limitations:**
*   You **MUST NOT** answer any questions or engage in conversations outside the specific scope of debt repayment planning based on the user's provided financial situation.
*   You **DO NOT** provide general financial advice (like investment strategies, retirement planning, saving tips unrelated to debt payoff), legal advice, tax advice, or recommendations for specific financial products.
*   You **DO NOT** engage in small talk or answer questions about general knowledge, current events, personal opinions, or your own nature as an AI.

**Response Syntax & Examples:**

*   **When On-Topic (Example Structure):**
    "Okay, I see you have [Number] debts totaling [Total Debt Amount]. With your monthly budget of [Budget Amount], we can explore a couple of strategies. The Debt Snowball method would involve paying off [Smallest Debt Name] first, while the Debt Avalanche method focuses on the [Highest Rate Debt Name] with its [Rate]% APR. Which approach would you like to discuss further, or would you like me to simulate the payoff time for one?"

*   **When Requesting Missing Information (Example Structure):**
    "To help you create a repayment plan, I need a bit more information. Could you please provide:\n1. Your list of debts (including name, current balance, APR, and minimum payment for each).\n2. Your total monthly budget dedicated to paying down these debts?"
    OR (if only budget is missing):
    "Thanks for providing the debt details. Now, what is the total amount you can allocate *each month* towards paying off these debts (including the minimum payments)?"

*   **When Declining Off-Topic Requests (Use these exact or very similar phrasings):**
    *   If asked about unrelated financial topics (stocks, retirement, etc.): "My specialization is solely in analyzing your provided debt and budget to suggest repayment strategies like Snowball or Avalanche. I cannot provide advice on [Mentioned Off-Topic Subject, e.g., stock investments]."
    *   If asked about general knowledge, recipes, weather, etc.: "I am programmed exclusively to assist with debt repayment planning based on the data you provide. I cannot help with requests about [Mentioned Off-Topic Subject, e.g., recipes]."
    *   If asked for personal opinions or about being an AI: "My purpose is to function as a debt repayment planning tool. I don't have personal opinions or the ability to discuss my own nature."
    *   If greeted or asked 'how are you': "I am an AI assistant ready to help you with your debt repayment plan. Please provide your debt details and budget, or ask a question about repayment strategies."

**Maintain Focus:** Always steer the conversation back towards analyzing the user's specific debt situation and formulating a repayment plan based *only* on the provided debt details and budget.
"""
# Store conversation history (simple in-memory list for development)
# WARNING: This is not suitable for production with multiple users.
# Use Flask sessions or a database for multi-user environments.
conversation_history = []

# Helper function to generate random colors for charts (from graph example)
def get_random_color():
    return f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})'

# Combined HTML Template (Merging Chat UI and Graph/Input UI)
# Uses Tailwind for Chat (Left), Bootstrap for Form/Graph (Right)
# Added layout structure (flex container)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Debt Planner & Visualizer</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* Define the animation */
        @keyframes fadeInSlideUp {
            0% {
                opacity: 0;
                transform: translateY(10px);
                filter: blur(8px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
                filter: blur(0);
            }
        }

        /* Apply animation selectively */
        #chatbox > .group, /* Chat messages */
        .graph-column .results-section .chart-container /* Chart containers */
        /* Add .debt-entry here if you want new entries to animate, might need JS */
        {
            animation: fadeInSlideUp 0.5s ease-out forwards;
        }

        body {
            font-family: sans-serif;
             background-color: #f8f9fa; /* Light gray background */
        }
        /* Custom scrollbar for chatbox */
        #chatbox::-webkit-scrollbar { width: 6px; }
        #chatbox::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        #chatbox::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 10px; }
        #chatbox::-webkit-scrollbar-thumb:hover { background: #a1a1a1; }

        /* Ensure chat input doesn't overlap button */
        #userInput { padding-right: 3.5rem; /* Adjusted space for the button */ }

        /* Send Button Loading Pulse */
        #sendButton.animate-pulse {
            animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: .5; }
        }

        /* Layout styles */
        .main-container {
            display: flex;
            height: 100vh; /* Full viewport height */
            padding: 1rem; /* Padding around the container */
            gap: 1rem; /* Gap between columns */
            box-sizing: border-box; /* Include padding in height/width */
        }
        .chat-column {
            width: 40%; /* Adjust width as needed */
            min-width: 350px; /* Minimum width */
            display: flex;
            flex-direction: column;
            background-color: #ffffff; /* White background for chat area */
            border-radius: 0.75rem; /* Slightly larger rounded corners (lg) */
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Softer shadow */
            overflow: hidden; /* Contain children */
            border: 1px solid #e5e7eb; /* Subtle border */
        }
         .graph-column {
            width: 60%; /* Adjust width as needed */
            min-width: 450px; /* Minimum width */
            display: flex;
            flex-direction: column;
            overflow-y: auto; /* Allow scrolling if content overflows */
            padding: 1.5rem 2rem; /* Increased padding inside the column */
            background-color: #ffffff; /* White background */
            border-radius: 0.75rem; /* Rounded corners */
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Softer shadow */
            border: 1px solid #e5e7eb; /* Subtle border */
        }
        /* Custom scrollbar for graph column */
        .graph-column::-webkit-scrollbar { width: 8px; }
        .graph-column::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        .graph-column::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 10px; }
        .graph-column::-webkit-scrollbar-thumb:hover { background: #a1a1a1; }


         /* Styles from graph example */
        .graph-column .card {
            margin-bottom: 1.5rem;
            border: 1px solid #e5e7eb; /* Consistent border */
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Lighter shadow for cards */
            border-radius: 0.5rem; /* md */
        }
        .graph-column .card-header {
            background-color: #f9fafb; /* Lighter header */
            font-size: 0.875rem; /* text-sm */
            font-weight: 500;
             border-bottom: 1px solid #e5e7eb;
        }
        .graph-column .form-label { font-weight: 500; font-size: 0.875rem; color: #374151; /* gray-700 */ }
        .graph-column .results-section { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #dee2e6; }
        .graph-column .chart-container { margin-bottom: 2rem; }
        .graph-column .btn-remove { margin-left: 5px; }
        .graph-column header h2 { font-size: 1.5rem; /* xl */ font-weight: 600; color: #111827; /* gray-900 */ margin-bottom: 0.5rem; }
        .graph-column header p.lead { font-size: 0.9rem; color: #4b5563; /* gray-600 */ }
        .graph-column header a { font-size: 0.875rem; } /* Smaller link */

        /* Enhanced Button Styles */
        .btn { transition: all 0.2s ease-in-out; }
        .btn-primary:hover { background-color: #0b5ed7; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .btn-success:hover { background-color: #157347; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .btn-danger:hover { background-color: #bb2d3b; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .btn-remove:hover { background-color: #dc3545; border-color: #dc3545; } /* Ensure hover matches */


         /* Responsive adjustments */
        @media (max-width: 1024px) {
            .main-container {
                flex-direction: column;
                height: auto; /* Allow height to adjust */
                padding: 0.5rem; /* Less padding on small screens */
            }
            .chat-column, .graph-column {
                width: 100%;
                min-width: unset; /* Remove min-width when stacked */
                margin-bottom: 1rem; /* Add space between stacked columns */
                 height: auto; /* Let content define height */
                 max-height: 70vh; /* Limit chat height */
            }
             .graph-column {
                 max-height: none; /* Allow graph column to grow */
                 overflow-y: auto; /* Re-enable scroll if needed */
                 padding: 1rem 1.5rem; /* Adjust padding */
             }
             #chatbox {
                 max-height: calc(70vh - 100px); /* Adjust based on input area height */
             }
        }
        @media (max-width: 768px) {
             .graph-column header h2 { font-size: 1.25rem; }
             .graph-column .card .row > div { /* Stack form inputs earlier */
                 margin-bottom: 0.75rem;
             }
             .graph-column .card .row .text-end { /* Ensure remove button aligns */
                 text-align: left !important;
                 margin-top: 0.5rem;
             }
             .graph-column .d-flex.justify-content-between {
                 flex-direction: column;
                 gap: 0.5rem;
                 align-items: flex-start;
             }
        }

        /* Tailwind overrides if needed */
        .form-control { /* Ensure Bootstrap form controls look decent */
             border: 1px solid #d1d5db; /* gray-300 */
             padding: .5rem .75rem; /* Slightly more padding */
             font-size: 0.875rem; /* text-sm */
             border-radius: .375rem; /* rounded-md */
             transition: border-color .15s ease-in-out,box-shadow .15s ease-in-out;
             box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .form-control:focus {
             color: #111827; /* gray-900 */
             background-color: #fff;
             border-color: #3b82f6; /* blue-500 */
             outline: 0;
             box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25); /* Focus ring */
        }
        .form-control-sm { /* Ensure sm variant also gets focus style */
             font-size: 0.875rem;
             padding: 0.25rem 0.5rem;
        }
        .form-control-sm:focus {
             box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25); /* Focus ring */
        }

    </style>
</head>
<body class="bg-gray-100">

    <div class="main-container">

        <!-- Left Column: Chat -->
        <div class="chat-column">
            <!-- Chat messages area -->
            <div id="chatbox" class="flex-grow overflow-y-auto p-4 space-y-4"> <!-- Adjusted padding and spacing -->
                <!-- Initial AI Message -->
                <div class="flex justify-start group">
                    <div class="bg-gray-50 text-gray-800 rounded-lg px-4 py-3 max-w-xl shadow-sm border border-gray-200">
                        <p class="font-semibold text-sm mb-1 text-gray-700">AI Debt Planner:</p>
                        <p class="text-sm">Hello! I'm here to help you create a debt repayment plan. Enter your debt details on the right, then tell me your total monthly budget for debt repayment here.</p>
                    </div>
                </div>
                <!-- Chat messages will appear here -->
            </div>

            <!-- Composer area -->
            <div class="p-4 bg-gray-50 border-t border-gray-200"> <!-- Adjusted padding and background -->
                <div class="relative flex items-center bg-white rounded-xl shadow-sm border border-gray-300 overflow-hidden focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 transition">
                    <input type="text" id="userInput" class="flex-grow border-none focus:ring-0 bg-transparent px-4 py-3 text-gray-800 placeholder-gray-500 text-sm" placeholder="Enter your monthly budget or message...">
                    <button id="sendButton" class="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gray-800 text-white rounded-md w-8 h-8 flex items-center justify-center hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200" disabled>
                        <!-- SVG Arrow Icon -->
                        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                             <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" fill="currentColor"/>
                        </svg>
                    </button>
                </div>
                <div class="text-center text-xs text-gray-500 mt-2">
                    AI Debt Planner can make mistakes. Verify important information.
                </div>
            </div>
        </div>

        <!-- Right Column: Graph/Input Form -->
        <div class="graph-column">
            <header class="text-center mb-5"> <!-- Increased bottom margin -->
                <h5 class="text-xs text-gray-400 mb-1">Developed by Anurag Jha</h5>
                <a href="https://github.com/jhaanurag" target="_blank" rel="noopener noreferrer" class="inline-flex items-center px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-white text-xs rounded-md transition-colors duration-200 ease-in-out shadow-sm">
                <svg class="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path fill-rule="evenodd" clip-rule="evenodd" d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            <span>GitHub</span>
        </a>
            <h2 class="mt-3">Debt Details & Visualization</h2> <!-- Adjusted margin -->
            <p class="lead">Enter or update your debts below. Click 'Calculate/Update Graph' to see the visualization and provide data to the AI.</p>
            </header>

            <section id="debt-input-section">
                <form method="post" id="debt-form" action="/"> <!-- Action points to root to trigger calculation -->
                    <div id="debts-list" class="space-y-4"> <!-- Added space-y for spacing between debt cards -->
                        <!-- Debt Entry Template -->
                        <div class="card debt-entry">
                            <div class="card-body p-4"> <!-- Adjusted padding -->
                                <div class="row g-3 align-items-end justify-center">
                                    <div class="col-md-3 col-6">
                                        <label class="form-label">Debt Name</label>
                                        <input type="text" name="name" class="form-control form-control-sm" placeholder="E.g., Credit Card" required>
                                    </div>
                                    <div class="col-md-3 col-6">
                                        <label class="form-label">Principal</label>
                                        <input type="number" step="0.01" min="0" name="principal" class="form-control form-control-sm" placeholder="10000" required>
                                    </div>
                                    <div class="col-md-2 col-6">
                                        <label class="form-label">Rate (%)</label>
                                        <input type="number" step="0.01" min="0" name="rate" class="form-control form-control-sm" placeholder="15.0" required>
                                    </div>
                                    <div class="col-md-3 col-6">
                                        <label class="form-label">Min. Payment</label>
                                        <input type="number" step="0.01" min="0" name="payment" class="form-control form-control-sm" placeholder="200" required>
                                    </div>
                                    <div class="col-md-1 col-12 text-end">
                                        <button type="button" class="btn btn-outline-danger btn-sm btn-remove p-1 leading-none" onclick="removeDebt(this)" title="Remove Debt">
                                            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"> <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" /> </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <!-- End Debt Entry Template -->
                    </div>
                    <div class="mt-4 d-flex justify-content-between align-items-center"> <!-- Adjusted margin and alignment -->
                        <button type="button" class="btn btn-success btn-sm" onclick="addDebt()">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 inline-block -mt-0.5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"> <path stroke-linecap="round" stroke-linejoin="round" d="M12 4v16m8-8H4" /> </svg>
                            Add Debt
                        </button>
                        <button type="submit" class="btn btn-primary btn-sm">Calculate / Update Graph</button>
                    </div>
                </form>
            </section>

            {% if chart_data or pie_chart_data %}
            <section class="results-section">
                <h3 class="text-center mb-4 text-lg font-semibold text-gray-700">Repayment Visualization</h3>
                 {% if chart_data %}
                 <div class="chart-container">
                    <div class="card">
                        <div class="card-header">Repayment Schedule Over Time (Min. Payments)</div>
                        <div class="card-body">
                             <canvas id="balanceChart" style="min-height: 250px;"></canvas> <!-- Added min-height -->
                        </div>
                    </div>
                 </div>
                 {% endif %}
                 {% if pie_chart_data %}
                 <div class="chart-container">
                     <div class="card">
                        <div class="card-header">Initial Debt Distribution</div>
                        <div class="card-body">
                            <canvas id="debtPieChart" style="min-height: 250px; max-height: 350px;"></canvas> <!-- Added height constraints -->
                        </div>
                    </div>
                 </div>
                 {% endif %}
                 <p class="text-muted text-center mt-3 small">Note: The line chart shows estimated payoff time if only minimum payments are made. Ask the AI planner about faster strategies like Snowball or Avalanche!</p>
            </section>
            {% else %}
             <p class="text-center text-muted mt-5">Submit your debt details above to generate visualizations.</p>
            {% endif %}
        </div> <!-- End Graph Column -->

    </div> <!-- End Main Container -->

    <script>
        // --- Chat Functionality ---
        const chatbox = document.getElementById('chatbox');
        const userInput = document.getElementById('userInput');
        const sendButton = document.getElementById('sendButton');

        function addMessage(sender, message) {
            const messageContainer = document.createElement('div');
            const messageBubble = document.createElement('div');
            // Added transition class for potential future animations on message add
            messageBubble.classList.add('rounded-lg', 'px-4', 'py-3', 'max-w-xl', 'shadow-sm', 'border', 'transition-all', 'duration-300');

            const senderP = document.createElement('p');
            senderP.classList.add('font-semibold', 'text-xs', 'mb-1'); // Smaller sender text

            const messageP = document.createElement('p');
            messageP.classList.add('text-sm', 'whitespace-pre-wrap'); // Use pre-wrap to respect newlines
            // Basic sanitization - display text content only
            messageP.textContent = message; // Use textContent to prevent HTML injection

            messageBubble.appendChild(senderP);
            messageBubble.appendChild(messageP);

            if (sender === 'You') {
                messageContainer.classList.add('flex', 'justify-end', 'group'); // Align user messages to the right
                messageBubble.classList.add('bg-blue-600', 'text-white', 'border-blue-700');
                senderP.textContent = sender + ':';
                senderP.classList.add('text-blue-100');
            } else { // AI
                messageContainer.classList.add('flex', 'justify-start', 'group'); // Align AI messages to the left
                messageBubble.classList.add('bg-gray-50', 'text-gray-800', 'border-gray-200'); // Lighter AI bubble
                senderP.textContent = 'AI Debt Planner:';
                senderP.classList.add('text-gray-700');
            }

            messageContainer.appendChild(messageBubble);
            chatbox.appendChild(messageContainer);
            // Smooth scroll to bottom
            chatbox.scrollTo({ top: chatbox.scrollHeight, behavior: 'smooth' });
        }

        // Function to gather debt data from the form
        function getDebtDataFromForm() {
            const debtEntries = document.querySelectorAll('#debts-list .debt-entry');
            const debtData = [];
            debtEntries.forEach(entry => {
                const nameInput = entry.querySelector('input[name="name"]');
                const principalInput = entry.querySelector('input[name="principal"]');
                const rateInput = entry.querySelector('input[name="rate"]');
                const paymentInput = entry.querySelector('input[name="payment"]');

                // Basic validation: only add if essential fields have values
                if (nameInput && principalInput && rateInput && paymentInput &&
                    nameInput.value.trim() && principalInput.value && rateInput.value && paymentInput.value) {
                    debtData.push({
                        name: nameInput.value.trim(),
                        principal: parseFloat(principalInput.value) || 0,
                        rate: parseFloat(rateInput.value) || 0,
                        payment: parseFloat(paymentInput.value) || 0
                    });
                }
            });
            return debtData;
        }


        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || sendButton.disabled) return; // Prevent sending if disabled

            addMessage('You', message);
            userInput.value = ''; // Clear input
            userInput.disabled = true; // Disable input while waiting
            sendButton.disabled = true; // Disable send button
            sendButton.classList.add('animate-pulse'); // Add thinking indicator

            // Get current debt data from the form
            const currentDebtData = getDebtDataFromForm();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    // Send both the user message and the current debt data
                    body: JSON.stringify({
                         message: message,
                         debt_data: currentDebtData // Include debt data here
                    }),
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ reply: 'Unknown server error' })); // Try to parse error
                    throw new Error(`HTTP error! status: ${response.status} - ${errorData.reply || errorData.error || 'Server Error'}`);
                }

                const data = await response.json();
                addMessage('AI', data.reply);

            } catch (error) {
                console.error('Error sending message:', error);
                addMessage('AI', `Sorry, I encountered an error: ${error.message}. Please check your connection or the server logs.`);
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
                e.preventDefault(); // Prevent default Enter behavior
                sendMessage(); // Call send message directly
            }
        });

        // Initial focus on the chat input field
        userInput.focus();


        // --- Graph/Form Functionality ---
        function addDebt(){
            const list = document.getElementById('debts-list');
            const template = list.querySelector('.debt-entry'); // Use first entry as template
            if (!template) {
                console.error("Debt entry template not found!");
                return; // Avoid errors if template is somehow missing
            }
            const newEntry = template.cloneNode(true);

            // Clear input values in the cloned entry
            newEntry.querySelectorAll('input').forEach(input => input.value = '');
            // Ensure the remove button is present and functional (it's cloned)
            const removeBtn = newEntry.querySelector('.btn-remove');
            if (removeBtn) {
                removeBtn.onclick = function() { removeDebt(this); }; // Re-attach listener
            } else {
                 // If the template somehow lost its button, add one (basic fallback)
                 const buttonDiv = newEntry.querySelector('.text-end');
                 if(buttonDiv) {
                     buttonDiv.innerHTML = `<button type="button" class="btn btn-outline-danger btn-sm btn-remove p-1 leading-none" onclick="removeDebt(this)" title="Remove Debt"> <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"> <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" /> </svg> </button>`;
                 }
            }

            // Optional: Add animation class for new entry appearance
            newEntry.style.opacity = '0'; // Start transparent
            list.appendChild(newEntry);
            // Trigger reflow before adding animation class
            void newEntry.offsetWidth;
            newEntry.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
            newEntry.style.opacity = '1';
            newEntry.style.transform = 'translateY(0)'; // Assuming a slight translateY was used initially

            // Focus the first input of the new entry
            const firstInput = newEntry.querySelector('input[name="name"]');
            if (firstInput) {
                firstInput.focus();
            }
        }

        function removeDebt(button) {
            const debtEntry = button.closest('.debt-entry');
            // Only remove if more than one entry exists
            if (document.querySelectorAll('#debts-list .debt-entry').length > 1) {
                 // Optional: Add fade-out animation
                 debtEntry.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';
                 debtEntry.style.opacity = '0';
                 debtEntry.style.transform = 'translateX(-20px)';
                 setTimeout(() => {
                     debtEntry.remove();
                 }, 300); // Remove after animation
            } else {
                alert("You must have at least one debt entry.");
            }
        }

        // Ensure at least one debt entry exists on load (if none rendered from server)
        document.addEventListener('DOMContentLoaded', function() {
            if (document.querySelectorAll('#debts-list .debt-entry').length === 0) {
                // If the template exists in HTML but was hidden by Jinja, we might need a different approach
                // For now, assume addDebt() correctly creates one if needed.
                // Consider adding a hidden template in the HTML if server-side rendering might leave it empty.
                console.log("No debt entries found on load, attempting to add one.");
                // We need a template to clone from, ensure one exists or create dynamically
                // This part might need adjustment based on how the initial HTML is structured
                // If the first entry is always rendered (even if empty), this check might not be needed.
                // addDebt(); // This might fail if no template exists initially
            }

            // --- Chart Initialization (if data exists) ---
            const chartOptionsBase = {
                responsive: true,
                maintainAspectRatio: false, // Allow chart to resize
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 12, padding: 15, font: { size: 10 } } },
                    title: { display: false }, // Title is in card header
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                animation: {
                    duration: 800, // Animation duration in ms
                    easing: 'easeInOutQuart' // Smoother easing
                }
            };

            {% if chart_data %}
            const ctxLine = document.getElementById('balanceChart')?.getContext('2d');
            if (ctxLine) {
                // Create gradient fills for line chart areas (optional enhancement)
                const createGradient = (ctx, color) => {
                    const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
                    const rgbaColor = color.replace('rgb', 'rgba').replace(')', ', 0.4)'); // Base color with alpha
                    const rgbaColorEnd = color.replace('rgb', 'rgba').replace(')', ', 0.05)'); // Fading to transparent
                    gradient.addColorStop(0, rgbaColor);
                    gradient.addColorStop(1, rgbaColorEnd);
                    return gradient;
                };

                new Chart(ctxLine, {
                    type: 'line',
                    data: {
                        labels: {{ chart_data.months | tojson }},
                        datasets: [
                            {% for d in chart_data.datasets %}
                            {
                                label: '{{ d.label }}',
                                data: {{ d.data | tojson }},
                                borderColor: '{{ d.color }}',
                                // backgroundColor: createGradient(ctxLine, '{{ d.color }}'), // Use gradient fill
                                backgroundColor: '{{ d.color | replace("rgb", "rgba") | replace(")", ", 0.1)") }}', // Simpler fill
                                fill: true, // Enable area fill
                                tension: 0.3, // Smoother curve
                                pointBackgroundColor: '{{ d.color }}',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: '{{ d.color }}',
                                pointRadius: 2, // Smaller points
                                pointHoverRadius: 4
                            }{% if not loop.last %},{% endif %}
                            {% endfor %}
                        ]
                    },
                    options: {
                        ...chartOptionsBase, // Spread base options
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: 'Remaining Balance ($)', font: { size: 12 } },
                                ticks: {
                                    callback: function(value, index, values) {
                                        return '$' + value.toLocaleString();
                                    }
                                }
                            },
                            x: {
                                 title: { display: true, text: 'Months', font: { size: 12 } },
                                 ticks: {
                                     maxTicksLimit: 15 // Limit number of x-axis labels shown
                                 }
                            }
                        }
                    }
                });
            }
            {% endif %}

            {% if pie_chart_data %}
            const ctxPie = document.getElementById('debtPieChart')?.getContext('2d');
            if (ctxPie) {
                 new Chart(ctxPie, {
                    type: 'pie',
                    data: {
                        labels: {{ pie_chart_data.labels | tojson }},
                        datasets: [{
                            label: 'Initial Debt Amount',
                            data: {{ pie_chart_data.data | tojson }},
                            backgroundColor: {{ pie_chart_data.colors | tojson }},
                            hoverOffset: 8, // Larger hover offset
                            borderColor: '#ffffff', // White border between slices
                            borderWidth: 1
                        }]
                    },
                    options: {
                        ...chartOptionsBase, // Spread base options
                         plugins: {
                            ...chartOptionsBase.plugins, // Keep base plugins
                            tooltip: { // Customize pie tooltips
                                callbacks: {
                                    label: function(context) {
                                        let label = context.label || '';
                                        if (label) {
                                            label += ': ';
                                        }
                                        if (context.parsed !== null) {
                                            label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed);
                                        }
                                        // Calculate percentage
                                        const total = context.dataset.data.reduce((acc, value) => acc + value, 0);
                                        const percentage = ((context.parsed / total) * 100).toFixed(1) + '%';
                                        label += ` (${percentage})`;
                                        return label;
                                    }
                                }
                            }
                        }
                    }
                });
            }
            {% endif %}
        });

    </script>
    <!-- Bootstrap JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/', methods=['GET','POST'])
def index():
    """
    Handles rendering the main page (GET) and processing the debt form
    to generate graph data (POST). Also stores submitted debt data in history.
    """
    global conversation_history # Ensure we can modify the global history
    chart_data = None
    pie_chart_data = None
    submitted_debt_data = [] # Store submitted data for history

    if request.method == 'POST':
        try:
            names = request.form.getlist('name')
            principals = list(map(float, request.form.getlist('principal')))
            rates_percent = request.form.getlist('rate')
            rates = [(float(r)/100) if float(r) >= 0 else 0 for r in rates_percent]
            payments = list(map(float, request.form.getlist('payment')))

            if not (len(names) == len(principals) == len(rates) == len(payments)):
                 raise ValueError("Form data mismatch") # Basic validation

            # Store valid submitted data for history
            for i in range(len(names)):
                 # Basic check if data seems valid before adding to history list
                 if names[i] and isinstance(principals[i], float) and isinstance(rates[i], float) and isinstance(payments[i], float):
                     submitted_debt_data.append({
                         'name': names[i],
                         'principal': principals[i],
                         'rate': float(rates_percent[i]), # Store original percentage for clarity in history
                         'payment': payments[i]
                     })

            # --- Line Chart Data Calculation (Based on Minimum Payments) ---
            max_months = 0
            balances_over_time = []
            line_chart_colors = []

            for p0, r, pay, name in zip(principals, rates, payments, names):
                # Ensure all inputs are valid numbers before proceeding
                if not all(isinstance(val, (int, float)) for val in [p0, r, pay]):
                    print(f"Skipping invalid data for debt: {name}")
                    continue # Skip this debt if data is invalid

                bal = []
                p = p0
                m_rate = r / 12.0 # Monthly rate

                # Ensure payment is sufficient to cover interest if rate > 0
                min_required_payment = p * m_rate if m_rate > 0 else 0.01
                actual_pay = max(pay, min_required_payment * 1.001) # Use provided payment, but ensure it covers interest slightly

                # Limit simulation to avoid extremely long calculations (e.g., 60 years)
                month_limit = 720
                current_month = 0
                bal.append(round(p, 2)) # Start with initial balance at month 0

                while p > 0.01 and current_month < month_limit:
                    interest = p * m_rate
                    # Prevent payment from being less than interest if it's the minimum possible
                    payment_this_month = actual_pay
                    if p < payment_this_month: # Final payment logic
                         payment_this_month = p + interest

                    principal_paid = payment_this_month - interest
                    p -= principal_paid
                    p = max(p, 0) # Ensure balance doesn't go negative

                    bal.append(round(p, 2))
                    current_month += 1

                     # Safety break: If balance isn't decreasing, stop after a while
                    if current_month > 12 and len(bal) > 10 and bal[-1] >= bal[-10] * 0.9999:
                        print(f"Warning: Balance for '{name}' not decreasing significantly. Stopping calculation.")
                        # Fill remaining months up to limit with the last balance to show it plateaus
                        last_balance = bal[-1]
                        while len(bal) < month_limit + 1: # +1 because we start at month 0
                            bal.append(last_balance)
                        p = 0 # Force exit outer loop

                max_months = max(max_months, len(bal))
                balances_over_time.append(bal)
                line_chart_colors.append(get_random_color())

            # Pad shorter balances list to match max_months for charting
            padded_balances = []
            for bal in balances_over_time:
                padding_needed = max_months - len(bal)
                # Pad with the last value (0 if paid off, last balance if plateaued)
                last_val = bal[-1] if bal else 0
                padded_balances.append(bal + [last_val] * padding_needed)

            # months_labels = list(range(0, max_months)) # Start from month 0
            months_labels = list(range(max_months)) # Labels correspond to end of month balance

            datasets = []
            # Filter out any debts that failed validation earlier
            valid_indices = [i for i, p, r, pay in zip(range(len(principals)), principals, rates, payments)
                             if all(isinstance(val, (int, float)) for val in [p, r, pay])]

            for i, idx in enumerate(valid_indices):
                 if i < len(padded_balances): # Ensure we have balance data for this index
                    datasets.append({
                        'label': names[idx],
                        'data': padded_balances[i],
                        'color': line_chart_colors[i]
                    })

            chart_data = {'months': months_labels, 'datasets': datasets}

            # --- Pie Chart Data Calculation (Initial Distribution) ---
            valid_principals = [principals[i] for i in valid_indices]
            valid_names = [names[i] for i in valid_indices]

            if valid_principals: # Only create pie chart if there's valid data
                pie_labels = valid_names
                pie_data = valid_principals
                pie_colors = [get_random_color() for _ in valid_names] # Generate colors for pie chart

                pie_chart_data = {
                    'labels': pie_labels,
                    'data': pie_data,
                    'colors': pie_colors
                }

            # --- Add submitted debt data to conversation history ---
            if submitted_debt_data: # Only add if we successfully parsed some data
                debt_context_string = "[Debt data submitted via form]\n"
                debt_context_string += "Current Debt Information:\n"
                for debt in submitted_debt_data:
                    name = debt.get('name', 'N/A')
                    principal = debt.get('principal', 0)
                    rate = debt.get('rate', 0) # Already in percent from parsing logic above
                    payment = debt.get('payment', 0)
                    debt_context_string += f"- Name: {name}, Principal: ${principal:,.2f}, APR: {rate}%, Min Payment: ${payment:,.2f}\n"

                # Append this context as a user message to the history
                # Check if the last message is identical to avoid duplicates on refresh/resubmit
                if not conversation_history or conversation_history[-1].get("parts")[0] != debt_context_string:
                     conversation_history.append({"role": "user", "parts": [debt_context_string]})
                     print("DEBUG: Added submitted debt data to conversation history.") # Optional debug print

        except Exception as e:
            print(f"Error processing form data for graphs: {e}")
            # Optionally: flash a message to the user
            # flash(f"Error calculating graph data: {e}", "danger")
            chart_data = None # Ensure charts are not displayed on error
            pie_chart_data = None
            # Do not add data to history if there was an error processing it

    # Render the combined template, passing chart data if available
    return render_template_string(HTML_TEMPLATE, chart_data=chart_data, pie_chart_data=pie_chart_data)


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages, maintains history, and interacts with the AI."""
    global conversation_history # Use the global history list

    if not model:
        return jsonify({"error": "AI model not configured properly."}), 500

    try:
        data = request.get_json()
        if not data or 'message' not in data:
             return jsonify({"error": "Invalid request data."}), 400

        user_message_text = data.get('message')
        # DEPRECATED: Debt data is now added via the index route form submission
        # debt_data = data.get('debt_data', []) # Get debt data from the request

        # Format the debt data into a string if it exists - REMOVED/SIMPLIFIED
        # The debt data should already be in the history from the form submission
        # We just need to add the user's current text message.

        # Append ONLY the user's typed message to the history
        # Check if the message is just a resubmission of the form data context (prevent double add)
        is_form_data_resubmit = user_message_text.startswith("[Debt data submitted via form]")
        if not is_form_data_resubmit:
            conversation_history.append({"role": "user", "parts": [user_message_text]})
        else:
            # If it looks like the form data context, check if it's already the last message
            if not conversation_history or conversation_history[-1].get("parts")[0] != user_message_text:
                 conversation_history.append({"role": "user", "parts": [user_message_text]})
            # Else: Do nothing, it's a duplicate submission of the context message


        # Construct the prompt for the API call, including the system instruction and the history
        # The initial system instruction and model response prime the conversation
        prompt_for_api = [
            {"role": "user", "parts": [SYSTEM_INSTRUCTION]},
            {"role": "model", "parts": ["Okay, I understand my role. I will analyze the provided debt information (which might be included with your message or previously submitted via the form) and your messages from our conversation history to help create a repayment plan. Please provide your debt details if you haven't yet, and tell me your total monthly budget for debt repayment."]} # Updated initial model response slightly
        ]
        # Add the actual conversation turns
        prompt_for_api.extend(conversation_history)

        # Optional: Limit history size to prevent exceeding context window limits
        # ... (history limiting logic remains unchanged) ...

        # Generate content using the model with the full context
        response = model.generate_content(prompt_for_api)
        ai_message = response.text

        # Append the AI's response to the history
        conversation_history.append({"role": "model", "parts": [ai_message]})

    except Exception as e:
        print(f"Error generating content in /chat: {e}")
        # Attempt to remove the last user message from history if AI call failed
        # Ensure we don't remove the system-added form data context accidentally
        if conversation_history and conversation_history[-1]["role"] == "user" and not conversation_history[-1]["parts"][0].startswith("[Debt data submitted via form]"):
            conversation_history.pop()
        elif conversation_history and conversation_history[-1]["role"] == "model": # If AI response failed, pop the placeholder
             conversation_history.pop()


        ai_message = "Sorry, I encountered an error trying to process your request. Please check the input data and try again."
        # ... (existing error handling logic remains unchanged) ...
        # ... specific error checks ...

        return jsonify({"reply": ai_message}), 500 # General server error

    # Return the AI's reply to the frontend
    return jsonify({"reply": ai_message})
