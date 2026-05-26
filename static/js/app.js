// Prediction
const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const loading = document.getElementById("loading");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    loading.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    loading.classList.add("hidden");

    document.getElementById("resultSection").classList.remove("hidden");
    document.getElementById("predictedClass").textContent = data.predicted_class;
    document.getElementById("confidence").textContent = data.confidence + "%";
    document.getElementById("previewImage").src = "data:image/png;base64," + data.image;

    const statusEl = document.getElementById("statusMessage");
    if (data.predicted_class === "Invalid") {
        statusEl.textContent = data.message;
        statusEl.className = "status warning";
    } else {
        statusEl.textContent = data.message;
        statusEl.className = "status success";
    }
});

// Chatbot
async function sendMessage() {
    const input = document.getElementById("chatInput");
    const chatBox = document.getElementById("chatBox");
    if (!input.value) return;

    chatBox.innerHTML += `<div><b>You:</b> ${input.value}</div>`;

    const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: input.value})
    });

    const data = await res.json();
    chatBox.innerHTML += `<div><b>Assistant:</b> ${data.response}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight; // scroll to bottom
    input.value = "";
}
