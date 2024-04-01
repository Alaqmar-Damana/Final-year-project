import React, { ChangeEvent, useEffect, useRef, useState } from "react";
import "./App.css";
import {
  MainContainer,
  MessageContainer,
  MessageHeader,
  MessageInput,
  MessageList,
  MinChatUiProvider,
} from "@minchat/react-chat-ui";
import MessageType from "@minchat/react-chat-ui/dist/types/MessageType";

function Greeting({ onNext }: { onNext: () => void }) {
  return (
    <div className="bg-slate-400 flex-1 h-full flex flex-col items-center justify-center gap-10">
      <div className="text-7xl font-bold bg-gradient-to-tr from-gray-500 via-slate-800 to-black bg-clip-text text-transparent shadow-black shadow-lg p-5 rounded-full">
        VidTalk
      </div>
      <div className="font-semibold text-2xl text-slate-800">
        It's time we have the talk, right?
      </div>
      <div
        className="bg-slate-800 p-7 py-4 text-white hover:bg-slate-700 rounded-lg shadow-xl shadow-slate-700 hover:shadow-slate-800"
        onClick={onNext}
      >
        Let's Start! ➡
      </div>
    </div>
  );
}

function ChatUI({ isLoading }: { isLoading: boolean }) {
  const [isProcessing, setIsProcessing] = useState(false);

  const [messages, setMessages] = useState<MessageType[]>([
    {
      text: "Hello, you can proceed with asking questions. I have processed the amazing video, with streams of profound knowledge to be bestowed upon mankind.",
      user: {
        id: "bot",
        name: "VidTalk Bot",
      },
    },
  ]);

  const sendMessage = async (text: string) => {
    setIsProcessing(true);
    setMessages((m) => [
      ...m,
      {
        text: text,
        user: {
          id: "user",
          name: "User",
        },
      },
    ]);
    setTimeout(() => {
      setMessages((m) => [
        ...m,
        {
          text: text.toUpperCase(),
          user: {
            id: "bot",
            name: "VidTalk Bot",
          },
        },
      ]);
      setIsProcessing(false);
    }, 100);
  };

  return (
    <MinChatUiProvider theme="#6ea9d7">
      <MainContainer style={{ height: "100%" }}>
        <MessageContainer>
          <MessageHeader
            showBack={false}
            mobileView={false}
            children={<div>VidTalk Chat</div>}
          />
          <MessageList
            currentUserId="user"
            messages={messages}
            loading={isLoading}
          />
          <MessageInput
            placeholder="Type message here"
            showSendButton={true}
            showAttachButton={false}
            onSendMessage={sendMessage}
            disabled={isProcessing || isLoading}
          />
        </MessageContainer>
      </MainContainer>
    </MinChatUiProvider>
  );
}

const fixYTLink = (link: string | null): string => {
  if (link && link.includes("youtu.be")) {
    return link.replace("youtu.be", "youtube.com/embed");
  }
  return link ?? "";
};

const EMOJI = {
  processing: "🔄",
  processed: "✅",
};

function Dashboard() {
  const [link, setLink] = useState("");

  const onChangeLink = (e: ChangeEvent<HTMLInputElement>) =>
    setLink(fixYTLink(e.target.value));

  useEffect(() => console.log("link::", link), [link]);

  const [processingStage, setProcessingStage] = useState<
    "unprocessed" | "processing" | "processed"
  >("unprocessed");

  const [processingProgress, setProcessingProgress] = useState(0);

  useEffect(() => console.log("x::", processingProgress), [processingProgress]);

  const interval = useRef<NodeJS.Timer>();

  const onProcess = () => {
    if (interval.current) {
      clearInterval(interval.current);
    }
    setProcessingStage("processing");
    setProcessingProgress(0);
    interval.current = setInterval(() => {
      setProcessingProgress((p) => p + 1);
    }, 1000);
  };

  useEffect(() => {
    if (processingProgress >= 100) {
      clearInterval(interval.current);
      setProcessingStage("processed");
    }
  }, [processingProgress]);

  return (
    <div className="flex-1 h-full flex flex-row bg-slate-200">
      <div className="w-2/5 h-full flex flex-col justify-start items-start p-10 gap-5">
        <div className="text-2xl font-semibold">
          Enter the link of a video to process:
        </div>
        {link && (
          <iframe
            title="yt-video"
            className="h-[30%] w-[60%] rounded-2xl"
            src={link}
          ></iframe>
        )}
        <input
          type="text"
          placeholder="Enter a YouTube link"
          className="border-4 outline-none focus:border-slate-800 text-slate-800 text-2xl p-2 w-full rounded-xl"
          onChange={onChangeLink}
        />
        <button
          className="bg-slate-800 p-7 py-4 text-white hover:bg-slate-700 rounded-full shadow-xl shadow-slate-700 hover:shadow-slate-800 font-bold text-xl"
          disabled={false}
          onClick={onProcess}
        >
          Process
        </button>
        {processingStage !== "unprocessed" && (
          <>
            <div className="mt-10 text-slate-800 text-2xl">
              {processingStage[0].toUpperCase() + processingStage.substring(1)}
              {processingStage === "processing" ? "..." : ""}
              {EMOJI[processingStage]}
            </div>
            <div className="w-full h-3 bg-slate-300 rounded-full overflow-hidden">
              <div
                className={`bg-green-800 h-full w-[${processingProgress}%]`}
              />
            </div>
          </>
        )}
      </div>
      <div className="w-3/5 h-full flex items-center justify-center">
        {processingStage !== "unprocessed" ? (
          <ChatUI isLoading={processingStage === "processing"} />
        ) : (
          <div className="text-slate-800 text-xl font-semibold">
            Once you process a video, you can proceed with asking questions.
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [appStage, setAppStage] = useState<"greeting" | "dashboard">(
    "greeting"
  );

  const openDashboard = () => setAppStage("dashboard");

  return appStage === "greeting" ? (
    <Greeting onNext={openDashboard} />
  ) : (
    <Dashboard />
  );
}

export default App;
