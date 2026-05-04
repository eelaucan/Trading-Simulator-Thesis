import type {
  ActionType,
  PlanImpactPayload,
  QuantityType,
  TradePlannerActionPayload,
  TradePlannerEventPayload,
  TradePlannerProps,
} from "./types.js";

const ACTION_LABELS: Record<ActionType, string> = {
  buy: "Buy shares",
  sell: "Sell shares",
  reduce: "Reduce a holding",
  set_stop: "Set a stop price",
  remove_stop: "Remove a stop price",
  hold: "Do nothing this week",
};

const ACTION_HELP: Record<ActionType, string> = {
  buy: "Choose a stock and decide how large your purchase should be.",
  sell: "Choose a stock and decide how much of it to sell.",
  reduce: "Trim an existing holding by a percentage.",
  set_stop: "Add a stop price that can schedule a forced sale if the market drops below it.",
  remove_stop: "Remove an active stop price from one of your holdings.",
  hold: "Make no new trading decision this week.",
};

const QUANTITY_LABELS: Record<QuantityType, string> = {
  shares: "By number of shares",
  notional_dollars: "By dollar amount",
  nav_fraction: "By fraction of portfolio value",
  close_all: "Sell the entire holding",
};

const EPSILON = 1e-12;

interface PlannerBridge {
  emit: (payload: TradePlannerEventPayload) => void;
  setFrameHeight: () => void;
}

interface FormState {
  actionType: ActionType;
  ticker: string;
  quantityType: QuantityType;
  quantityText: string;
  fractionText: string;
  stopPriceText: string;
  error: string | null;
}

export class TradePlannerApp {
  private readonly root: HTMLElement;
  private readonly bridge: PlannerBridge;
  private props: TradePlannerProps | null = null;
  private currentPlan: TradePlannerActionPayload[] = [];
  private eventCounter = 0;
  private formState: FormState = {
    actionType: "buy",
    ticker: "",
    quantityType: "shares",
    quantityText: "1",
    fractionText: "0.25",
    stopPriceText: "",
    error: null,
  };

  constructor(root: HTMLElement, bridge: PlannerBridge) {
    this.root = root;
    this.bridge = bridge;
  }

  public setProps(props: TradePlannerProps): void {
    this.props = props;
    this.currentPlan = props.current_batch.map((action) => ({ ...action }));
    this.syncFormState();
    this.render();
  }

  private render(): void {
    if (!this.props) {
      this.root.innerHTML = "<div class='tp-loading'>Loading planner…</div>";
      this.bridge.setFrameHeight();
      return;
    }

    const props = this.props;
    const remainingSlots = Math.max(
      0,
      props.max_actions_per_step - this.currentPlan.length,
    );
    const previewIsSynced =
      this.planSignature(this.currentPlan) === this.planSignature(props.current_batch);

    this.root.innerHTML = `
      <section class="tp-shell">
        <div class="tp-grid">
          <div class="tp-panel tp-panel--ticket">
            <div class="tp-kicker">Decision Ticket</div>
            <h2 class="tp-title">Place your decision for next week's open</h2>
            <p class="tp-subtitle">
              Build one action at a time, then review the full plan before you submit.
            </p>
            <div class="tp-badges">
              <span class="tp-badge">Week ${props.current_week_index + 1}</span>
              <span class="tp-badge">${remainingSlots} slot${remainingSlots === 1 ? "" : "s"} left</span>
            </div>
            <div class="tp-step-row">
              <span class="tp-step-pill">1. Choose action</span>
              <span class="tp-step-pill">2. Choose stock</span>
              <span class="tp-step-pill">3. Size trade</span>
              <span class="tp-step-pill">4. Add decision</span>
            </div>
            <div class="tp-form">
              ${this.renderForm(props, remainingSlots)}
            </div>
          </div>
          <div class="tp-panel tp-panel--review">
            <div class="tp-review-head">
              <div>
                <div class="tp-review-kicker">Weekly Plan</div>
                <h3 class="tp-review-title">Review before you submit</h3>
              </div>
              <div class="tp-review-count">${this.currentPlan.length}/${props.max_actions_per_step} actions</div>
            </div>
            ${this.renderPlanReview(props)}
            ${this.renderImpactPreview(props.plan_impact, previewIsSynced)}
            <div class="tp-submit-area">
              <div class="tp-submit-copy">
                The simulator checks your plan against hard rules. Any already scheduled forced sale executes first. Valid trades then move to next week's open.
              </div>
              <div class="tp-submit-actions">
                <button class="tp-button tp-button--ghost" id="tp-clear-plan" ${
                  this.currentPlan.length === 0 || props.disabled ? "disabled" : ""
                }>
                  Clear plan
                </button>
                <button class="tp-button tp-button--primary" id="tp-submit-plan" ${
                  props.disabled ? "disabled" : ""
                }>
                  Submit this week's decisions
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    `;

    this.attachListeners();
    this.bridge.setFrameHeight();
  }

  private renderForm(props: TradePlannerProps, remainingSlots: number): string {
    const actionType = this.formState.actionType;
    const tickerOptions = this.getTickerOptions(actionType, props);
    const disableAdd =
      remainingSlots <= 0 || tickerOptions.length === 0 || Boolean(props.disabled);

    const actionOptions = ([
      "buy",
      "sell",
      "reduce",
      "set_stop",
      "remove_stop",
    ] as ActionType[])
      .map(
        (value) =>
          `<option value="${value}" ${
            actionType === value ? "selected" : ""
          }>${escapeHtml(ACTION_LABELS[value])}</option>`,
      )
      .join("");

    const tickerField =
      actionType === "hold"
        ? ""
        : `
          <label class="tp-field">
            <span class="tp-label">Choose the stock</span>
            <select id="tp-ticker" ${tickerOptions.length === 0 ? "disabled" : ""}>
              ${tickerOptions
                .map(
                  (ticker) =>
                    `<option value="${escapeHtml(ticker)}" ${
                      this.formState.ticker === ticker ? "selected" : ""
                    }>${escapeHtml(ticker)}</option>`,
                )
                .join("")}
            </select>
            ${
              tickerOptions.length === 0
                ? `<span class="tp-field-note">${escapeHtml(this.emptyTickerMessage(actionType))}</span>`
                : this.renderTickerContext(this.formState.ticker, props)
            }
          </label>
        `;

    const sizingField = this.renderSizingFields(actionType);

    return `
      <label class="tp-field">
        <span class="tp-label">Choose your action</span>
        <select id="tp-action-type">
          ${actionOptions}
        </select>
        <span class="tp-field-note">${escapeHtml(ACTION_HELP[actionType])}</span>
      </label>
      ${tickerField}
      ${sizingField}
      ${
        this.formState.error
          ? `<div class="tp-inline-error">${escapeHtml(this.formState.error)}</div>`
          : ""
      }
      <button class="tp-button tp-button--secondary" id="tp-add-action" ${
        disableAdd ? "disabled" : ""
      }>
        Add this decision to your weekly plan
      </button>
    `;
  }

  private renderSizingFields(actionType: ActionType): string {
    if (actionType === "buy") {
      return `
        <label class="tp-field">
          <span class="tp-label">How would you like to size this purchase?</span>
          <select id="tp-quantity-type">
            ${(["shares", "notional_dollars", "nav_fraction"] as QuantityType[])
              .map(
                (value) =>
                  `<option value="${value}" ${
                    this.formState.quantityType === value ? "selected" : ""
                  }>${escapeHtml(QUANTITY_LABELS[value])}</option>`,
              )
              .join("")}
          </select>
        </label>
        <label class="tp-field">
          <span class="tp-label">${escapeHtml(this.quantityInputLabel(this.formState.quantityType))}</span>
          <input id="tp-quantity" type="number" min="0.0001" step="0.01" value="${escapeHtml(
            this.formState.quantityText,
          )}" />
        </label>
      `;
    }

    if (actionType === "sell") {
      const quantitySelector = `
        <label class="tp-field">
          <span class="tp-label">How would you like to size this sale?</span>
          <select id="tp-quantity-type">
            ${(["shares", "notional_dollars", "close_all"] as QuantityType[])
              .map(
                (value) =>
                  `<option value="${value}" ${
                    this.formState.quantityType === value ? "selected" : ""
                  }>${escapeHtml(QUANTITY_LABELS[value])}</option>`,
              )
              .join("")}
          </select>
        </label>
      `;
      if (this.formState.quantityType === "close_all") {
        return quantitySelector;
      }
      return `
        ${quantitySelector}
        <label class="tp-field">
          <span class="tp-label">${escapeHtml(this.quantityInputLabel(this.formState.quantityType))}</span>
          <input id="tp-quantity" type="number" min="0.0001" step="0.01" value="${escapeHtml(
            this.formState.quantityText,
          )}" />
        </label>
      `;
    }

    if (actionType === "reduce") {
      return `
        <label class="tp-field">
          <span class="tp-label">Enter the fraction to reduce</span>
          <input id="tp-fraction" type="number" min="0.0001" max="1" step="0.01" value="${escapeHtml(
            this.formState.fractionText,
          )}" />
          <span class="tp-field-note">Use 0.25 for 25% or 0.50 for half the position.</span>
        </label>
      `;
    }

    if (actionType === "set_stop") {
      return `
        <label class="tp-field">
          <span class="tp-label">Enter the stop price</span>
          <input id="tp-stop-price" type="number" min="0.01" step="0.01" value="${escapeHtml(
            this.formState.stopPriceText,
          )}" />
          <span class="tp-field-note">This does not trade immediately. It can schedule a forced sale if a later weekly low breaches the price.</span>
        </label>
      `;
    }

    return `
      <div class="tp-field tp-field--quiet">
        <span class="tp-label">No size entry is needed for this action.</span>
      </div>
    `;
  }

  private renderTickerContext(ticker: string, props: TradePlannerProps): string {
    if (!ticker) {
      return "";
    }
    const holding = props.holdings.find((item) => item.ticker === ticker);
    const currentClose = props.close_prices[ticker];
    const parts: string[] = [];
    if (holding && holding.shares > EPSILON) {
      parts.push(`You currently hold ${formatShares(holding.shares)}.`);
    }
    if (typeof currentClose === "number" && currentClose > 0) {
      parts.push(`Visible close: ${formatCurrency(currentClose)}.`);
    }
    if (holding && typeof holding.active_stop === "number") {
      parts.push(`Active stop: ${formatCurrency(holding.active_stop)}.`);
    }
    return parts.length > 0
      ? `<span class="tp-field-note">${escapeHtml(parts.join(" "))}</span>`
      : "";
  }

  private renderPlanReview(props: TradePlannerProps): string {
    if (this.currentPlan.length === 0) {
      const pendingCopy =
        props.pending_liquidations.length > 0
          ? "Earlier forced sales would still execute if they were already scheduled."
          : "If you submit now, your current portfolio simply carries into next week.";
      return `
        <div class="tp-empty-state">
          <div class="tp-empty-title">No actions added yet.</div>
          <div class="tp-empty-copy">${escapeHtml(pendingCopy)}</div>
        </div>
      `;
    }

    return `
      <div class="tp-plan-list">
        ${this.currentPlan
          .map((action, index) => {
            const detail = action.detail
              ? `<div class="tp-plan-detail">${escapeHtml(action.detail)}</div>`
              : "";
            return `
              <div class="tp-plan-row">
                <div class="tp-plan-copy">
                  <div class="tp-plan-index">Decision ${index + 1}</div>
                  <div class="tp-plan-summary">${escapeHtml(action.summary ?? summarizeAction(action))}</div>
                  ${detail}
                </div>
                <button class="tp-button tp-button--remove" data-remove-index="${index}">
                  Remove
                </button>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  }

  private renderImpactPreview(
    preview: PlanImpactPayload,
    previewIsSynced: boolean,
  ): string {
    if (this.currentPlan.length === 0) {
      return "";
    }

    if (!previewIsSynced) {
      return `
        <div class="tp-preview-card tp-preview-card--pending">
          Updating estimates from the visible prices…
        </div>
      `;
    }

    const largestWeight =
      typeof preview.projected_max_weight === "number"
        ? formatPercent(preview.projected_max_weight)
        : "N/A";

    const warnings = preview.warnings.length
      ? `
        <div class="tp-warning-box">
          <div class="tp-warning-title">Plan warnings</div>
          <ul>
            ${preview.warnings
              .map((item) => `<li>${escapeHtml(item)}</li>`)
              .join("")}
          </ul>
        </div>
      `
      : `
        <div class="tp-warning-box tp-warning-box--calm">
          This plan looks broadly feasible from the visible prices, although the simulator still applies hard rules when you submit.
        </div>
      `;

    const notes = preview.notes.length
      ? `<div class="tp-preview-notes">${preview.notes
          .map((note) => `<div>${escapeHtml(note)}</div>`)
          .join("")}</div>`
      : "";

    return `
      <div class="tp-preview-card">
        <div class="tp-preview-heading">Estimated plan impact</div>
        <div class="tp-metric-grid tp-metric-grid--primary">
          ${this.metricCard("Planned actions", String(this.currentPlan.length))}
          ${this.metricCard("Estimated spend", formatCurrency(preview.estimated_spend))}
          ${this.metricCard("Estimated costs", formatCurrency(preview.estimated_transaction_costs))}
          ${this.metricCard("Cash after plan", formatCurrency(preview.estimated_remaining_cash))}
        </div>
        <div class="tp-preview-caption">Estimated from currently visible prices only.</div>
        <div class="tp-metric-grid tp-metric-grid--secondary">
          ${this.metricCard("Estimated proceeds", formatCurrency(preview.estimated_proceeds))}
          ${this.metricCard("Positions after plan", String(preview.estimated_positions_after))}
          ${this.metricCard("Invested after plan", formatCurrency(preview.estimated_invested_after))}
          ${this.metricCard("Largest est. weight", largestWeight)}
        </div>
        ${warnings}
        ${notes}
      </div>
    `;
  }

  private metricCard(label: string, value: string): string {
    return `
      <div class="tp-metric-card">
        <div class="tp-metric-label">${escapeHtml(label)}</div>
        <div class="tp-metric-value">${escapeHtml(value)}</div>
      </div>
    `;
  }

  private attachListeners(): void {
    if (!this.props) {
      return;
    }

    const actionTypeInput = this.root.querySelector<HTMLSelectElement>("#tp-action-type");
    actionTypeInput?.addEventListener("change", (event) => {
      this.formState.actionType = (event.currentTarget as HTMLSelectElement)
        .value as ActionType;
      this.formState.error = null;
      this.syncFormState();
      this.render();
    });

    const tickerInput = this.root.querySelector<HTMLSelectElement>("#tp-ticker");
    tickerInput?.addEventListener("change", (event) => {
      this.formState.ticker = (event.currentTarget as HTMLSelectElement).value;
      this.render();
    });

    const quantityTypeInput =
      this.root.querySelector<HTMLSelectElement>("#tp-quantity-type");
    quantityTypeInput?.addEventListener("change", (event) => {
      this.formState.quantityType = (event.currentTarget as HTMLSelectElement)
        .value as QuantityType;
      this.formState.error = null;
      this.render();
    });

    const quantityInput = this.root.querySelector<HTMLInputElement>("#tp-quantity");
    quantityInput?.addEventListener("input", (event) => {
      this.formState.quantityText = (event.currentTarget as HTMLInputElement).value;
    });

    const fractionInput = this.root.querySelector<HTMLInputElement>("#tp-fraction");
    fractionInput?.addEventListener("input", (event) => {
      this.formState.fractionText = (event.currentTarget as HTMLInputElement).value;
    });

    const stopPriceInput = this.root.querySelector<HTMLInputElement>("#tp-stop-price");
    stopPriceInput?.addEventListener("input", (event) => {
      this.formState.stopPriceText = (event.currentTarget as HTMLInputElement).value;
    });

    const addButton = this.root.querySelector<HTMLButtonElement>("#tp-add-action");
    addButton?.addEventListener("click", () => {
      const action = this.buildActionFromForm();
      if (!action) {
        this.render();
        return;
      }
      this.currentPlan = [...this.currentPlan, action];
      this.emit("plan_change");
      this.syncFormState();
      this.render();
    });

    const clearButton = this.root.querySelector<HTMLButtonElement>("#tp-clear-plan");
    clearButton?.addEventListener("click", () => {
      this.currentPlan = [];
      this.formState.error = null;
      this.emit("plan_change");
      this.render();
    });

    const submitButton = this.root.querySelector<HTMLButtonElement>("#tp-submit-plan");
    submitButton?.addEventListener("click", () => {
      this.emit("submit");
    });

    this.root
      .querySelectorAll<HTMLButtonElement>("[data-remove-index]")
      .forEach((button) => {
        button.addEventListener("click", () => {
          const rawIndex = button.dataset.removeIndex;
          const index = rawIndex ? Number.parseInt(rawIndex, 10) : -1;
          if (Number.isNaN(index) || index < 0 || index >= this.currentPlan.length) {
            return;
          }
          this.currentPlan = this.currentPlan.filter((_, rowIndex) => rowIndex !== index);
          this.emit("plan_change");
          this.render();
        });
      });
  }

  private emit(eventType: "plan_change" | "submit"): void {
    this.eventCounter += 1;
    this.bridge.emit({
      event_id: `${Date.now()}-${this.eventCounter}`,
      event_type: eventType,
      actions: this.currentPlan.map((action) => ({ ...action })),
    });
  }

  private buildActionFromForm(): TradePlannerActionPayload | null {
    const actionType = this.formState.actionType;
    const ticker = this.formState.ticker || null;

    if (this.currentPlan.length >= (this.props?.max_actions_per_step ?? 0)) {
      this.formState.error = "This weekly plan is already full.";
      return null;
    }

    if (actionType !== "hold" && !ticker) {
      this.formState.error = "Choose a stock before adding this decision.";
      return null;
    }

    if (actionType === "buy") {
      const quantity = this.parsePositiveNumber(this.formState.quantityText);
      if (quantity === null) {
        this.formState.error = "Enter a purchase amount greater than zero.";
        return null;
      }
      return this.decorateAction({
        action_type: "buy",
        ticker,
        quantity,
        quantity_type: this.formState.quantityType,
      });
    }

    if (actionType === "sell") {
      if (this.formState.quantityType === "close_all") {
        return this.decorateAction({
          action_type: "sell",
          ticker,
          quantity_type: "close_all",
        });
      }
      const quantity = this.parsePositiveNumber(this.formState.quantityText);
      if (quantity === null) {
        this.formState.error = "Enter a sale amount greater than zero.";
        return null;
      }
      return this.decorateAction({
        action_type: "sell",
        ticker,
        quantity,
        quantity_type: this.formState.quantityType,
      });
    }

    if (actionType === "reduce") {
      const fraction = this.parsePositiveNumber(this.formState.fractionText);
      if (fraction === null || fraction > 1) {
        this.formState.error = "Enter a fraction between 0 and 1.";
        return null;
      }
      return this.decorateAction({
        action_type: "reduce",
        ticker,
        fraction,
      });
    }

    if (actionType === "set_stop") {
      const stopPrice = this.parsePositiveNumber(this.formState.stopPriceText);
      if (stopPrice === null) {
        this.formState.error = "Enter a stop price greater than zero.";
        return null;
      }
      return this.decorateAction({
        action_type: "set_stop",
        ticker,
        stop_price: stopPrice,
      });
    }

    return this.decorateAction({
      action_type: "remove_stop",
      ticker,
    });
  }

  private decorateAction(
    action: TradePlannerActionPayload,
  ): TradePlannerActionPayload {
    this.formState.error = null;
    return {
      ...action,
      summary: summarizeAction(action),
      detail: actionDetail(action),
    };
  }

  private syncFormState(): void {
    if (!this.props) {
      return;
    }

    const availableActions: ActionType[] = [
      "buy",
      "sell",
      "reduce",
      "set_stop",
      "remove_stop",
    ];
    if (!availableActions.includes(this.formState.actionType)) {
      this.formState.actionType = "buy";
    }

    const actionType = this.formState.actionType;
    const tickerOptions = this.getTickerOptions(actionType, this.props);
    if (tickerOptions.length === 0) {
      this.formState.ticker = "";
    } else if (!tickerOptions.includes(this.formState.ticker)) {
      this.formState.ticker = tickerOptions[0];
    }

    if (actionType === "buy" && this.formState.quantityType === "close_all") {
      this.formState.quantityType = "shares";
    }
    if (actionType === "sell" && this.formState.quantityType === "nav_fraction") {
      this.formState.quantityType = "shares";
    }
    if (actionType === "remove_stop") {
      this.formState.stopPriceText = "";
    }
  }

  private getTickerOptions(
    actionType: ActionType,
    props: TradePlannerProps,
  ): string[] {
    return props.ticker_options[actionType] ?? [];
  }

  private emptyTickerMessage(actionType: ActionType): string {
    if (actionType === "reduce") {
      return "You do not currently hold any stock that can be reduced.";
    }
    if (actionType === "set_stop") {
      return "You need an active holding before you can set a stop price.";
    }
    if (actionType === "remove_stop") {
      return "You do not currently have any active stop prices.";
    }
    return "No stock is currently available for this action.";
  }

  private quantityInputLabel(quantityType: QuantityType): string {
    if (quantityType === "shares") {
      return "How many shares?";
    }
    if (quantityType === "notional_dollars") {
      return "How many dollars?";
    }
    if (quantityType === "nav_fraction") {
      return "What fraction of your portfolio value? (0.10 = 10%)";
    }
    return "Amount";
  }

  private parsePositiveNumber(text: string): number | null {
    const value = Number.parseFloat(text);
    if (!Number.isFinite(value) || value <= 0) {
      return null;
    }
    return value;
  }

  private planSignature(actions: TradePlannerActionPayload[]): string {
    return JSON.stringify(
      actions.map((action) => ({
        action_type: action.action_type,
        ticker: action.ticker ?? null,
        quantity: action.quantity ?? null,
        quantity_type: action.quantity_type ?? null,
        fraction: action.fraction ?? null,
        stop_price: action.stop_price ?? null,
      })),
    );
  }
}

function summarizeAction(action: TradePlannerActionPayload): string {
  if (action.action_type === "buy") {
    if (action.quantity_type === "shares" && typeof action.quantity === "number") {
      return `Buy ${formatShares(action.quantity)} of ${action.ticker ?? ""}`.trim();
    }
    if (
      action.quantity_type === "notional_dollars" &&
      typeof action.quantity === "number"
    ) {
      return `Buy ${formatCurrency(action.quantity)} of ${action.ticker ?? ""}`.trim();
    }
    return `Buy ${formatPercent(action.quantity ?? 0)} of portfolio value in ${action.ticker ?? ""}`.trim();
  }
  if (action.action_type === "sell") {
    if (action.quantity_type === "close_all") {
      return `Sell all shares of ${action.ticker ?? ""}`.trim();
    }
    if (action.quantity_type === "shares" && typeof action.quantity === "number") {
      return `Sell ${formatShares(action.quantity)} of ${action.ticker ?? ""}`.trim();
    }
    return `Sell ${formatCurrency(action.quantity ?? 0)} of ${action.ticker ?? ""}`.trim();
  }
  if (action.action_type === "reduce") {
    return `Reduce ${action.ticker ?? ""} by ${formatPercent(action.fraction ?? 0, 0)}`.trim();
  }
  if (action.action_type === "set_stop") {
    return `Set a stop price on ${action.ticker ?? ""} at ${formatCurrency(action.stop_price ?? 0)}`.trim();
  }
  if (action.action_type === "remove_stop") {
    return `Remove the stop price from ${action.ticker ?? ""}`.trim();
  }
  return "Do nothing this week";
}

function actionDetail(action: TradePlannerActionPayload): string | null {
  if (action.action_type === "buy" && action.quantity_type === "nav_fraction") {
    return "This buy is sized as a share of portfolio value at the start of the week.";
  }
  if (action.action_type === "sell" && action.quantity_type === "close_all") {
    return "The simulator will try to fully close this holding.";
  }
  if (action.action_type === "set_stop") {
    return "If a later weekly low breaches this price, the simulator can schedule a forced sale.";
  }
  return null;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

function formatShares(value: number): string {
  const rounded = Math.round(value * 10000) / 10000;
  if (Math.abs(rounded - Math.round(rounded)) <= EPSILON) {
    const whole = Math.round(rounded);
    return `${whole.toLocaleString("en-US")} ${whole === 1 ? "share" : "shares"}`;
  }
  return `${rounded.toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 4,
  })} shares`;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
