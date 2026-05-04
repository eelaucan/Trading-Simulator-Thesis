import { TradePlannerApp } from "./TradePlanner.js";
import type { TradePlannerEventPayload, TradePlannerProps } from "./types.js";

const root = document.getElementById("root");
if (!root) {
  throw new Error("Trade planner root element was not found.");
}

const postToStreamlit = (
  type: string,
  payload: Record<string, unknown>,
): void => {
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type,
      ...payload,
    },
    "*",
  );
};

const app = new TradePlannerApp(root, {
  emit: (payload: TradePlannerEventPayload): void => {
    postToStreamlit("streamlit:setComponentValue", {
      value: payload,
      dataType: "json",
    });
  },
  setFrameHeight: (): void => {
    window.requestAnimationFrame(() => {
      const height = Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
      );
      postToStreamlit("streamlit:setFrameHeight", { height });
    });
  },
});

const onRender = (event: MessageEvent): void => {
  const payload = event.data;
  if (!payload || payload.type !== "streamlit:render") {
    return;
  }
  const nextProps: TradePlannerProps = {
    ...(payload.args as TradePlannerProps),
    disabled: Boolean(payload.disabled),
  };
  app.setProps(nextProps);
};

window.addEventListener("message", onRender);
window.addEventListener("load", () => {
  postToStreamlit("streamlit:componentReady", { apiVersion: 1 });
});
