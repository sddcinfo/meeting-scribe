// Meeting Scribe — Q&A panel (pure module).
//
// Streaming chat-style Q&A against a finalized meeting transcript.
// Called from the finalize-summary feature after the summary card
// mounts; the qa panel hangs off the same modal. Posts to
// /api/meetings/<mid>/ask and streams an SSE-shaped response into
// the assistant bubble.
//
// Pure module — no top-level side effects. The bare function
// ``initQaPanel(meetingId)`` is consumed by finalize-summary and by
// the standalone meeting-review page.

const API = "";
const _enc = encodeURIComponent;

export function initQaPanel(meetingId) {
  const input = document.getElementById("qa-input");
  const sendBtn = document.getElementById("qa-send");
  const messagesEl = document.getElementById("qa-messages");
  if (!input || !sendBtn || !messagesEl) return;

  let qaInFlight = false;

  const submitQuestion = async () => {
    const question = input.value.trim();
    if (!question || qaInFlight) return;

    // Show user message
    const userMsg = document.createElement("div");
    userMsg.className = "qa-msg qa-msg-user";
    userMsg.textContent = question;
    messagesEl.appendChild(userMsg);
    input.value = "";
    qaInFlight = true;
    sendBtn.disabled = true;
    sendBtn.textContent = "...";

    // Create assistant message bubble
    const assistantMsg = document.createElement("div");
    assistantMsg.className = "qa-msg qa-msg-assistant";
    assistantMsg.textContent = "";
    messagesEl.appendChild(assistantMsg);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
      const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Request failed" }));
        assistantMsg.textContent = err.error || "Request failed";
        assistantMsg.classList.add("qa-msg-error");
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "chunk") {
              assistantMsg.textContent += data.text;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            } else if (data.type === "error") {
              assistantMsg.textContent = data.text;
              assistantMsg.classList.add("qa-msg-error");
            }
          } catch {}
        }
      }
    } catch (e) {
      assistantMsg.textContent = "Connection error: " + e.message;
      assistantMsg.classList.add("qa-msg-error");
    } finally {
      qaInFlight = false;
      sendBtn.disabled = false;
      sendBtn.textContent = "Ask";
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  };

  sendBtn.addEventListener("click", submitQuestion);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuestion();
    }
  });
}
