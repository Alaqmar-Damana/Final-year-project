import React, { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import axios from "axios";

function Greeting({ onNext }: { onNext: () => void }) {
  return (
    <div className="bg-slate-400 flex-1 h-full flex flex-col items-center justify-center gap-10">
      <div className="text-7xl font-bold bg-gradient-to-tr from-gray-500 via-slate-800 to-black bg-clip-text text-transparent shadow-black shadow-lg p-5 rounded-full">
        VidTalk
      </div>
      <div className="font-semibold text-2xl text-slate-800">
        Empowering Effortless Video Engagement and Information Discovery
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
    try {
      const response = await axios.get(`http://localhost:5000/get_answer?query=${text}`);
      setMessages((m) => [
        ...m,
        {
          text: response.data.answer,
          user: {
            id: "bot",
            name: "VidTalk Bot",
          },
        },
      ]);
      setIsProcessing(false);
    } catch (error) {
      setMessages((m) => [
        ...m,
        {
          text: "Couldn't comprehend, probably a network error. Please try again after a few minutes, or check your API key.",
          user: {
            id: "bot",
            name: "VidTalk Bot",
          },
        },
      ]);
      setIsProcessing(false);
    }
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
            showTypingIndicator={isProcessing}
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

const processYTLinkToEmbedLink = (link: string | null): string => {
  if (link && link.includes("youtu.be")) {
    return link.replace("youtu.be", "youtube.com/embed");
  } else if(link && link.includes("watch?v=")) {
    return link.replace("watch?v=", "embed/");
  }
  return link ?? "";
};

const EMOJI = {
  processing: "🔄",
  processed: "✅",
};

function Dashboard() {
  const [link, setLink] = useState("");

  const embedLink = useMemo(() => processYTLinkToEmbedLink(link), [link]);

  const onChangeLink = (e: ChangeEvent<HTMLInputElement>) =>
    setLink(e.target.value ?? "");

  useEffect(() => console.log("link::", link), [link]);

  const [processingStage, setProcessingStage] = useState<
    "unprocessed" | "processing" | "processed"
  >("unprocessed");

  const [processingProgress, setProcessingProgress] = useState(0);

  useEffect(() => console.log("x::", processingProgress), [processingProgress]);

  const interval = useRef<NodeJS.Timer>();

  const [duration, setDuration] = useState<number>(0);

  const checkIfProcessed = useCallback(async () => {
    const response = await axios.get('http://localhost:5000/is_processing');
    const isProcessed = response.data.processed === 1;
    if(isProcessed) {
      setProcessingStage("processed");
      setProcessingProgress(100);
      clearInterval(interval.current);
    }
  }, []);

  const updateProcessingProgress = useCallback(() => {
    if (processingProgress < 99) {
      setProcessingProgress(p => (p + 1 <= 99 ? p + 1 : 99))
    } else {
      setProcessingProgress(99)
    }
  }, [processingProgress]);

  useEffect(() => {
    if(!duration || processingProgress !== 0) return;
    console.log("duration", duration)
    interval.current = setInterval(() => {
      updateProcessingProgress()
      checkIfProcessed()
    }, (duration * 1000 * 3) / 100);
  }, [duration, processingProgress])

  const onProcess = async () => {
    if (interval.current) {
      clearInterval(interval.current);
    }
    setProcessingStage("processing");
    setProcessingProgress(0);
    const response = await axios.get(`http://localhost:5000/update_vid_link?link=${link}`);
    console.log("response::", response.data.length)
    setDuration(response.data.length);
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
        {embedLink && (
          <iframe
            title="yt-video"
            className="h-[30%] w-[60%] rounded-2xl"
            src={embedLink}
          />
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
        {processingStage !== "unprocessed" && duration && (
          <>
            <div className="mt-10 text-slate-800 text-2xl">
              {processingStage[0].toUpperCase() + processingStage.substring(1)}
              {processingStage === "processing" ? `... (${processingProgress} %) ` : ""}
              {EMOJI[processingStage]}
            </div>
            <div className="w-full h-3 bg-slate-300 rounded-full overflow-hidden">
              <div
                className={`bg-green-800 h-full ${processingProgress ? `w-[${processingProgress}%]` : 'w-0'}`}
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
