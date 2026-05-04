import { TradePlannerApp } from "./TradePlanner.js";
const root = document.getElementById("root");
if (!root) {
  throw new Error("Trade planner root element was not found.");
}
const postToStreamlit = (type, payload) => {
  window.parent.postMessage({
    isStreamlitMessage: true,
    type,
    ...payload
  }, "*");
};
const app = new TradePlannerApp(root, {
  emit: (payload) => {
    postToStreamlit("streamlit:setComponentValue", {
      value: payload,
      dataType: "json"
    });
  },
  setFrameHeight: () => {
    window.requestAnimationFrame(() => {
      const height = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
      postToStreamlit("streamlit:setFrameHeight", { height });
    });
  }
});
const onRender = (event) => {
  const payload = event.data;
  if (!payload || payload.type !== "streamlit:render") {
    return;
  }
  const nextProps = {
    ...payload.args,
    disabled: Boolean(payload.disabled)
  };
  app.setProps(nextProps);
};
window.addEventListener("message", onRender);
window.addEventListener("load", () => {
  postToStreamlit("streamlit:componentReady", { apiVersion: 1 });
});

