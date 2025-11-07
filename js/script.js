// ===================== ELEMENT SELECTION =====================
const messageform = document.querySelector(".prompt--form");
const inputbox = document.querySelector(".prompt--form--input");
const chathistorycontainer = document.querySelector(".chats");
const clearChatbutton = document.getElementById("deleteButton");
const sendbutton = document.getElementById("sendbutton");

// ===================== TOAST ALERTS =====================
const toastbox = document.querySelector(".toastalert");
function showtoast(msg, type = "Info") {
  let toast = document.createElement("div");
  toast.classList.add("toast", type);
  toast.innerHTML = msg;
  toastbox.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("hidealert");
    setTimeout(() => toast.remove(), 400);
  }, 3000);
}

// ===================== CHAT HANDLING =====================
let currentusermsg = null;
let isgeneratingresponse = false;

const createchatmessageelement = (htmlcontent, ...classes) => {
  const messageelement = document.createElement("div");
  messageelement.classList.add("message", ...classes);
  messageelement.innerHTML = htmlcontent;
  return messageelement;
};

function scrollToBottom() {
  setTimeout(() => {
    chathistorycontainer.scrollTo({
      top: chathistorycontainer.scrollHeight,
      behavior: "smooth",
    });
  }, 100);
}

// Typing animation
const showtypingeffect = (rawtext, htmltext, messageElement) => {
  const chars = rawtext.split("");
  let idx = 0;
  messageElement.innerText = "";

  const typingInterval = setInterval(() => {
    messageElement.innerText += chars[idx++];
    scrollToBottom();
    if (idx >= chars.length) {
      clearInterval(typingInterval);
      messageElement.innerHTML = marked.parse(rawtext);
      hljs.highlightAll();
      isgeneratingresponse = false;
    }
  }, 10);
};

// ===================== FASTAPI REQUEST =====================
const requestapiresponse = async (incomingmessageelemet) => {
  const messagetextElement = incomingmessageelemet.querySelector(".message--text");
  const userInput = currentusermsg;
  const formData = new FormData();
  formData.append("prompt_form_input", userInput);

  try {
    const response = await fetch("http://127.0.0.1:8000/process_user_input", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const responsetext = await response.text();
    const parsedapiresponse = marked.parse(responsetext);
    showtypingeffect(responsetext, parsedapiresponse, messagetextElement);
  } catch (error) {
    messagetextElement.innerText = "⚠️ " + error.message;
    messagetextElement.closest(".message").classList.add("message--error");
    isgeneratingresponse = false;
    scrollToBottom();
  } finally {
    incomingmessageelemet.classList.remove("message--loading");
    scrollToBottom();
  }
};

// ===================== DISPLAY + EVENTS =====================
const displayloadinganimation = () => {
  const loadinghtml = `
    <div class="message--content">
      <img class="message__avatar" src="images/sairam.jpeg" alt="avatar">
      <p class="message--text"></p>
      <div class="cssload-loader">
        <div>${"<div class='cssload-dot'></div>".repeat(10)}</div>
        <div>${"<div class='cssload-dotb'></div>".repeat(10)}</div>
      </div>
    </div>`;

  const loadingmessageelement = createchatmessageelement(
    loadinghtml,
    "message--incoming",
    "message--loading"
  );

  chathistorycontainer.appendChild(loadingmessageelement);
  scrollToBottom();
  requestapiresponse(loadingmessageelement);
};

const handleoutgoingmessage = () => {
  currentusermsg = inputbox.value.trim();
  if (!currentusermsg || isgeneratingresponse) return;

  isgeneratingresponse = true;
  const outgoinghtml = `
    <div class="message--content">
      <img class="message__avatar" src="images/favicon.png" alt="avatar">
      <p class="message--text">${currentusermsg}</p>
    </div>`;
  const outgoingmessageelement = createchatmessageelement(outgoinghtml, "message--outgoing");
  chathistorycontainer.appendChild(outgoingmessageelement);

  inputbox.value = "";
  scrollToBottom();
  setTimeout(displayloadinganimation, 400);
};

// ===================== BUTTON HANDLERS =====================
clearChatbutton.addEventListener("click", () => {
  if (confirm("Clear all messages?")) chathistorycontainer.innerHTML = "";
});

inputbox.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    if (!inputbox.value.trim()) {
      showtoast("⚠️ Enter a prompt.", "Warning");
      return;
    }
    handleoutgoingmessage();
  }
});

sendbutton.onclick = () => {
  if (!inputbox.value.trim()) {
    showtoast("⚠️ Enter a prompt.", "Warning");
    return;
  }
  handleoutgoingmessage();
};

// ===================== INIT =====================
showtoast("✅ RAG Assistant Ready! Ask your query now.");
scrollToBottom();