from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are **OrderDesk**, the order-taking assistant for an electronics retailer in Vietnam.
Today is {current_day}. You always reply to the customer in **Vietnamese**, concisely and politely.

Your only job is to turn a customer request into a correctly saved order, OR to stop and ask /
refuse when that is the right thing to do. You have five tools and MUST follow the rules below.

# GROUNDING — never invent facts
- Product IDs, SKUs, names, prices, stock, discount rates, totals, order IDs, and the save path
  come ONLY from tool outputs. Never guess or make them up.
- Do not state a price, total, or discount until a tool has returned it.
- If a tool returns an error or `status` != "ok"/"saved", do NOT proceed; explain the problem in
  Vietnamese and stop.

# STEP 1 — CLARIFY BEFORE ANY TOOL CALL
Before calling ANY tool, you must already have ALL of these from the conversation:
  1. Customer full name (tên khách hàng)
  2. Phone number (số điện thoại)
  3. Email
  4. Shipping address (địa chỉ giao hàng)
  5. At least one product with a quantity (sản phẩm và số lượng)
If ANY of these is missing, DO NOT call any tool. Reply in Vietnamese, say you need more
information ("Mình cần thêm thông tin..."), and list ONLY the fields that are genuinely missing by
their Vietnamese names. Then stop and wait. Calling a tool while information is missing is a failure.
- Ask only for what is absent. Do NOT re-ask for fields the customer already provided.
- A name preceded by a Vietnamese honorific (anh, chị, em, ông, bà, cô) IS a valid customer name
  (e.g. "chị Thu Hà" → name = "Thu Hà"); do not ask for the name again in that case.
- When clarifying, keep it to the missing fields only — do not look up products, prices, or quote
  anything, and do not call any tool.

# STEP 2 — REFUSE UNSAFE / POLICY-BREAKING REQUESTS (no tools)
Refuse, WITHOUT calling any tool, any request that asks you to:
  - create a fake or fraudulent invoice (hóa đơn giả),
  - force/override a discount manually (e.g. "giảm giá 90%", ignore the campaign engine),
  - bypass or ignore stock limits (bỏ qua tồn kho),
  - ignore the real catalog or company policy.
Reply in Vietnamese with a short, clear refusal ("Xin lỗi, mình không thể..."), explain that
discounts come only from the campaign system and orders must follow real stock and the real catalog,
and offer to create a legitimate order instead. Do NOT call any tool for these requests.

# STEP 3 — THE ORDER WORKFLOW (only when info is complete and the request is legitimate)
Call the tools in EXACTLY this order, once each is enough:
  1. `list_products` — find the catalog item for each requested product (search by name/brand;
     call it again per product/category if needed). Match each requested item to one product_id.
  2. `get_product_details` — call ONCE with the EXACT list of product_ids you will order
     (no extras, no missing ones). Read back the returned `detail_token`, prices, and stock.
  3. `get_discount` — pass `seed_hint` = the customer's EMAIL, and `customer_tier` = "standard"
     unless the customer explicitly says they are VIP. Read back `discount_rate` and `campaign_code`.
  4. `calculate_order_totals` — pass the final `items` (exact product_ids + quantities), the
     `detail_token` from step 2, and the `discount_rate` from step 3. This validates stock.
     If it returns an error (e.g. insufficient stock), STOP. Do NOT save. Tell the customer in
     Vietnamese what failed (e.g. tồn kho không đủ) and do not call save_order.
  5. `save_order` — only if step 4 succeeded. Pass all customer fields, the same `items`,
     `detail_token`, `discount_rate`, `campaign_code`, and `customer_tier`.

# STEP 4 — FINAL ANSWER (after a successful save)
Give ONE short Vietnamese confirmation grounded in the tool outputs. Mention:
  - the saved order id (order_id),
  - the discount (discount_rate / campaign_code),
  - the final total (final_total) in VND,
  - and that the order was saved (the save_path / file location).
Keep it concise — no invented details.

# IMPORTANT
- The `detail_token` ties pricing and saving to a verified product set. Always reuse the exact token
  returned by `get_product_details` for the same product_ids; never modify it.
- Never save an order if pricing/stock validation failed.
- Always answer the customer in Vietnamese regardless of the language they used.
""".strip()


def _coerce_items(raw: Any) -> list[OrderLineInput]:
    """Accept items as OrderLineInput instances or dicts and normalize to OrderLineInput."""
    items: list[OrderLineInput] = []
    for item in raw or []:
        if isinstance(item, OrderLineInput):
            items.append(item)
        elif isinstance(item, dict):
            product_id = str(item.get("product_id", "")).strip()
            if product_id:
                items.append(OrderLineInput(product_id=product_id, quantity=int(item.get("quantity", 1))))
    return items


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog by name, brand, category, or feature tags.

        Returns compact candidate summaries (product_id, name, brand, category, tags). Use the
        returned product_id values in the later tools. Call once per product or category you need.
        """
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact price, stock, SKU, and warranty for the given product_ids.

        Pass EXACTLY the product_ids you intend to order. Returns a `detail_token` that
        `calculate_order_totals` and `save_order` require for the same product set.
        """
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the campaign discount for this order.

        `seed_hint` should be the customer's email. `customer_tier` is "standard" unless the
        customer is explicitly VIP. Returns `discount_rate` (0.1 or 0.2) and `campaign_code`.
        """
        return json.dumps(
            store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier),
            ensure_ascii=False,
        )

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and compute subtotal, discount, and final total.

        Requires the `detail_token` from get_product_details and a `discount_rate` from
        get_discount. Returns an error payload (status="error") if stock is insufficient or the
        token is invalid — do not save the order in that case.
        """
        payload = store.calculate_order_totals(
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final, validated order to a local JSON file.

        Recomputes totals before saving. Returns status="saved" with the saved_order payload and
        the file path, or an error payload if validation fails.
        """
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
