import { RawData, WebSocket } from "ws";
import functions from "./functionHandlers";

interface Session {
  twilioConn?: WebSocket;
  modelConn?: WebSocket;
  streamSid?: string;
  saved_config?: any;
  lastAssistantItem?: string;
  responseStartTimestamp?: number;
  latestMediaTimestamp?: number;
  openAIApiKey?: string;
}

const sessions: Record<string, Session> = {};
let frontendConn: WebSocket | undefined = undefined;

export function handleCallConnection(ws: WebSocket, openAIApiKey: string) {
  ws.on("message", (data) => handleTwilioMessage(ws, data, openAIApiKey));
  ws.on("error", ws.close);
  ws.on("close", () => {
    for (const [sid, session] of Object.entries(sessions)) {
      if (session.twilioConn === ws) {
        cleanupConnection(session.modelConn);
        cleanupConnection(session.twilioConn);
        delete sessions[sid];
        console.log("Closed Twilio connection", sid);
      }
    }
  });
}

export function handleFrontendConnection(ws: WebSocket) {
  frontendConn = ws;
  ws.on("message", (data) => handleFrontendMessage(data));
  ws.on("close", () => {
    if (frontendConn === ws) {
      cleanupConnection(frontendConn);
      frontendConn = undefined;
    }
  });
}

async function handleFunctionCall(
  session: Session,
  item: { name: string; arguments: string }
) {
  const fnDef = functions.find((f) => f.schema.name === item.name);
  if (!fnDef)
    return JSON.stringify({ error: `No handler for function: ${item.name}` });

  let args;
  try {
    args = JSON.parse(item.arguments);
  } catch {
    return JSON.stringify({ error: "Invalid JSON arguments" });
  }

  try {
    const result = await fnDef.handler(args);
    return result;
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

function handleTwilioMessage(ws: WebSocket, data: RawData, apiKey: string) {
  const msg = parseMessage(data);
  if (!msg) return;

  if (msg.event === "start") {
    const sid = msg.start.streamSid;
    console.log("New Twilio connection", sid);
    sessions[sid] = {
      twilioConn: ws,
      streamSid: sid,
      openAIApiKey: apiKey,
      latestMediaTimestamp: 0,
    };
    tryConnectModel(sid);
  }

  const session = Object.values(sessions).find((s) => s.twilioConn === ws);
  if (!session) return;

  switch (msg.event) {
    case "media":
      session.latestMediaTimestamp = msg.media.timestamp;
      if (isOpen(session.modelConn)) {
        jsonSend(session.modelConn, {
          type: "input_audio_buffer.append",
          audio: msg.media.payload,
        });
      }
      break;
    case "close":
      cleanupConnection(session.twilioConn);
      cleanupConnection(session.modelConn);
      if (!frontendConn) delete sessions[session.streamSid!];
      break;
  }
}

function handleFrontendMessage(data: RawData) {
  const msg = parseMessage(data);
  if (!msg) return;

  // Optional: Sende an alle Sessions gleichzeitig (falls nötig)
  for (const session of Object.values(sessions)) {
    if (isOpen(session.modelConn)) {
      jsonSend(session.modelConn, msg);
    }
    if (msg.type === "session.update") {
      session.saved_config = msg.session;
    }
  }
}

function tryConnectModel(sid: string) {
  const session = sessions[sid];
  if (!session || session.modelConn || !session.openAIApiKey) return;

  session.modelConn = new WebSocket(
    "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17",
    {
      headers: {
        Authorization: `Bearer ${session.openAIApiKey}`,
        "OpenAI-Beta": "realtime=v1",
      },
    }
  );

  session.modelConn.on("open", () => {
    jsonSend(session.modelConn, {
      type: "session.update",
      session: {
        modalities: ["text", "audio"],
        turn_detection: { type: "server_vad" },
        voice: "ash",
        input_audio_transcription: { model: "whisper-1" },
        input_audio_format: "g711_ulaw",
        output_audio_format: "g711_ulaw",
        ...session.saved_config,
      },
    });
  });

  session.modelConn.on("message", (data) => handleModelMessage(session, data));
  session.modelConn.on("error", () => closeModel(session));
  session.modelConn.on("close", () => closeModel(session));
}

function handleModelMessage(session: Session, data: RawData) {
  const event = parseMessage(data);
  if (!event) return;

  jsonSend(frontendConn, event);

  switch (event.type) {
    case "input_audio_buffer.speech_started":
      handleTruncation(session);
      break;
    case "response.audio.delta":
      if (session.twilioConn && session.streamSid) {
        if (session.responseStartTimestamp === undefined) {
          session.responseStartTimestamp = session.latestMediaTimestamp || 0;
        }
        if (event.item_id) session.lastAssistantItem = event.item_id;

        jsonSend(session.twilioConn, {
          event: "media",
          streamSid: session.streamSid,
          media: { payload: event.delta },
        });

        jsonSend(session.twilioConn, {
          event: "mark",
          streamSid: session.streamSid,
        });
      }
      break;
    case "response.output_item.done": {
      const { item } = event;
      if (item.type === "function_call") {
        handleFunctionCall(session, item).then((output) => {
          if (session.modelConn) {
            jsonSend(session.modelConn, {
              type: "conversation.item.create",
              item: {
                type: "function_call_output",
                call_id: item.call_id,
                output: JSON.stringify(output),
              },
            });
            jsonSend(session.modelConn, { type: "response.create" });
          }
        });
      }
      break;
    }
  }
}

function handleTruncation(session: Session) {
  if (
    !session.lastAssistantItem ||
    session.responseStartTimestamp === undefined
  )
    return;

  const elapsedMs =
    (session.latestMediaTimestamp || 0) - (session.responseStartTimestamp || 0);
  const audio_end_ms = elapsedMs > 0 ? elapsedMs : 0;

  if (isOpen(session.modelConn)) {
    jsonSend(session.modelConn, {
      type: "conversation.item.truncate",
      item_id: session.lastAssistantItem,
      content_index: 0,
      audio_end_ms,
    });
  }

  if (session.twilioConn && session.streamSid) {
    jsonSend(session.twilioConn, {
      event: "clear",
      streamSid: session.streamSid,
    });
  }

  session.lastAssistantItem = undefined;
  session.responseStartTimestamp = undefined;
}

function closeModel(session: Session) {
  cleanupConnection(session.modelConn);
  session.modelConn = undefined;
  if (!session.twilioConn && !frontendConn) {
    delete sessions[session.streamSid!];
  }
}

function cleanupConnection(ws?: WebSocket) {
  if (isOpen(ws)) ws.close();
}

function parseMessage(data: RawData): any {
  try {
    return JSON.parse(data.toString());
  } catch {
    return null;
  }
}

function jsonSend(ws: WebSocket | undefined, obj: unknown) {
  if (!isOpen(ws)) return;
  ws.send(JSON.stringify(obj));
}

function isOpen(ws?: WebSocket): ws is WebSocket {
  return !!ws && ws.readyState === WebSocket.OPEN;
}
