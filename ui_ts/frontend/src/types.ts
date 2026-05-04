export type ActionType =
  | "buy"
  | "sell"
  | "reduce"
  | "set_stop"
  | "remove_stop"
  | "hold";

export type QuantityType =
  | "shares"
  | "notional_dollars"
  | "nav_fraction"
  | "close_all";

export interface TradePlannerActionPayload {
  action_type: ActionType;
  ticker?: string | null;
  quantity?: number | null;
  quantity_type?: QuantityType | null;
  fraction?: number | null;
  stop_price?: number | null;
  summary?: string;
  detail?: string | null;
}

export interface HoldingRow {
  ticker: string;
  shares: number;
  average_cost: number;
  market_value: number;
  weight: number;
  active_stop?: number | null;
  current_close?: number | null;
}

export interface PendingLiquidationRow {
  ticker: string;
  triggered_by_low: number;
  stop_level: number;
  execution_week: number;
}

export interface PlanImpactPayload {
  estimated_spend: number;
  estimated_proceeds: number;
  estimated_transaction_costs: number;
  estimated_remaining_cash: number;
  estimated_positions_after: number;
  estimated_invested_after: number;
  projected_max_weight?: number | null;
  warnings: string[];
  notes: string[];
}

export interface TradePlannerProps {
  current_week_index: number;
  current_date: string;
  max_actions_per_step: number;
  remaining_action_slots: number;
  available_tickers: string[];
  current_cash: number;
  current_total_nav: number;
  current_batch: TradePlannerActionPayload[];
  holdings: HoldingRow[];
  active_stops: Record<string, number>;
  pending_liquidations: PendingLiquidationRow[];
  ticker_options: Record<string, string[]>;
  close_prices: Record<string, number>;
  plan_impact: PlanImpactPayload;
  disabled?: boolean;
}

export interface TradePlannerEventPayload {
  event_id: string;
  event_type: "plan_change" | "submit";
  actions: TradePlannerActionPayload[];
}

