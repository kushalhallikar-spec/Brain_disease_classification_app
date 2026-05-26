// ----------------------------
// Frontend JS for AI Chat + MRI
// ----------------------------

// Select DOM elements
const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const fileInput = document.getElementById("fileInput");
const chatBox = document.getElementById("chatBox");

// Your OpenAI API key (for local testing only)
const OPENAI_API_KEY = "YOUR_API_KEY_HERE";

// Send button click event
sendBtn.addEventListener("click", () => {
    const message = userInput.value.trim();
    if (message || (fileInput.files && fileInput.files.length > 0)) {
        sendMessage(message);
        userInput.value = "";
    }
});

// Send message function
async function sendMessage(userMessage) {
    try {
        // Display user message
        if (userMessage) displayMessage(userMessage, "user");

        // Show typing placeholder
        displayMessage("Typing...", "assistant", true);

        let botMessage = "";

        if (fileInput.files && fileInput.files.length > 0) {
            // ----------------------------
            // MRI Prediction
            // ----------------------------
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            let data;
            try {
                data = await response.json();
            } catch {
                data = { response: "Error: Unable to parse server response" };
            }
            botMessage = data.response || "Error: No response from server";

        } else {
            // ----------------------------
            // AI Medical Chat
            // ----------------------------
            const response = await fetch("https://api.openai.com/v1/chat/completions", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${OPENAI_API_KEY}`
                },
                body: JSON.stringify({
                    model: "gpt-4",
                    messages: [
                        { role: "system", content: "You are a helpful medical assistant." },
                        { role: "user", content: userMessage }
                    ],
                    temperature: 0.5
                })
            });

            let data;
            try {
                data = await response.json();
            } catch {
                data = { error: "Error: Unable to parse OpenAI response" };
            }

            if (data.error) {
                botMessage = data.error.message || "Error calling OpenAI API";
            } else {
                botMessage = data.choices[0].message.content.trim();
            }
        }

        // Remove typing placeholder
        const placeholder = document.getElementById("typingPlaceholder");
        if (placeholder) placeholder.remove();

        // Display bot message
        displayMessage(botMessage, "assistant");

    } catch (err) {
        console.error(err);
        const placeholder = document.getElementById("typingPlaceholder");
        if (placeholder) placeholder.remove();
        displayMessage("Error: Unable to contact server", "assistant");
    }
}

// Display messages in chat
function displayMessage(message, sender, isPlaceholder = false) {
    const messageElement = document.createElement("div");
    messageElement.className = sender === "user" ? "user-msg" : "assistant-msg";
    messageElement.textContent = message || "";
    if (isPlaceholder) messageElement.id = "typingPlaceholder";
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // auto scroll
}
