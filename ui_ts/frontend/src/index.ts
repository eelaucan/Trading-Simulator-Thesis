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

const standaloneProps: TradePlannerProps = {
  current_week_index: 3,
  current_date: "2026-05-04",
  max_actions_per_step: 5,
  remaining_action_slots: 5,
  available_tickers: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
  current_cash: 38250,
  current_total_nav: 100000,
  current_batch: [],
  holdings: [
    {
      ticker: "AAPL",
      shares: 75,
      average_cost: 185.4,
      market_value: 14520,
      weight: 0.1452,
      active_stop: 170,
      current_close: 193.6,
    },
    {
      ticker: "MSFT",
      shares: 40,
      average_cost: 410,
      market_value: 17180,
      weight: 0.1718,
      active_stop: null,
      current_close: 429.5,
    },
    {
      ticker: "NVDA",
      shares: 120,
      average_cost: 96.25,
      market_value: 30050,
      weight: 0.3005,
      active_stop: null,
      current_close: 250.42,
    },
  ],
  active_stops: {
    AAPL: 170,
  },
  pending_liquidations: [],
  ticker_options: {
    buy: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    sell: ["AAPL", "MSFT", "NVDA"],
    reduce: ["AAPL", "MSFT", "NVDA"],
    set_stop: ["AAPL", "MSFT", "NVDA"],
    remove_stop: ["AAPL"],
  },
  close_prices: {
    AAPL: 193.6,
    MSFT: 429.5,
    NVDA: 250.42,
    GOOGL: 174.25,
    AMZN: 184.8,
  },
  plan_impact: {
    estimated_spend: 0,
    estimated_proceeds: 0,
    estimated_transaction_costs: 0,
    estimated_remaining_cash: 38250,
    estimated_positions_after: 3,
    estimated_invested_after: 61750,
    projected_max_weight: 0.3005,
    warnings: [],
    notes: ["Public preview data is illustrative; Streamlit supplies live simulator state."],
  },
  disabled: false,
};

if (window.parent === window) {
  app.setProps(standaloneProps);
}

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
